"""PyTorch DistributedDataParallel adapter -- the one fully-implemented
distributed-training framework this pass. No extra dependency: DDP ships
inside `torch` already. Trains the same PyTorch/Image Classification combo
across processes using the gloo backend (CPU-friendly, works the same on
every device class in scope) with TCP rendezvous -- that rendezvous plus
the per-step gradient all-reduce traffic is exactly what gets captured.

Data loading goes through core/training_data.py's build_training_dataset()
(config.dataset, via the DATASETS/APPLICATIONS registries) rather than
hardcoding torchvision.datasets.CIFAR10 the way this adapter originally
did -- see core/config.py's FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES and
core/training_data.py for exactly which architecture/application/dataset
combinations that supports and why, and core/training_objectives.py for
the per-architecture loss (classification/reconstruction/denoising) this
adapter no longer hardcodes to CrossEntropyLoss.

torch/torchvision are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
either installed -- only actually running the DDP slice needs them.
"""
import os

from core.registry import ARCHITECTURES, DISTRIBUTED_FRAMEWORKS, FRAMEWORKS
from core.training_data import build_training_dataset
from core.training_objectives import get_trainable_module, get_training_step, set_trainable_module
from distributed_frameworks.base import DistributedFrameworkAdapter


def _init_process_group(config, rank):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", config.host)
    os.environ.setdefault("MASTER_PORT", str(config.port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=max(config.num_clients, 1))


def _build_loader(config):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = build_training_dataset(config, max(config.num_clients, 1) * config.num_requests)
    sampler = DistributedSampler(dataset, num_replicas=max(config.num_clients, 1), rank=dist.get_rank())

    # Same real BatchNorm2d/batch-size-1 crash documented in
    # fl_frameworks/flower_adapter.py's _partition_loader -- floor at 2,
    # drop any leftover partial batch of 1.
    return DataLoader(dataset, batch_size=max(config.batch_size, 2), sampler=sampler, drop_last=True)


def _train(config, logger, event_log, rank):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    _init_process_group(config, rank)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework, config)
        # Same real device-mismatch crash documented in
        # fl_frameworks/flower_adapter.py's fit() -- captured before
        # DistributedDataParallel wraps the model, since this loader's
        # tensors are plain CPU and need to match wherever the model
        # actually is (cuda when available).
        device = next(model.parameters()).device
        # architectures/ddpm.py's DDPM needs only its inner noise_predictor
        # wrapped, not the whole T-step sampling loop -- see
        # core/training_objectives.py's get_trainable_module() docstring.
        # Every other architecture wraps unchanged (get_trainable_module
        # just returns model itself for those).
        trainable = get_trainable_module(model, config.architecture)
        ddp_trainable = DistributedDataParallel(trainable)
        model = set_trainable_module(model, config.architecture, ddp_trainable)

        loader = _build_loader(config)
        optimizer = torch.optim.SGD(ddp_trainable.parameters(), lr=0.01)
        training_step = get_training_step(config.architecture)

        model.train()
        for round_index in range(config.num_rounds):
            total = 0
            for batch in loader:
                if total >= config.num_requests:
                    break
                optimizer.zero_grad()
                loss, n = training_step(model, batch, device)
                loss.backward()  # gradient all-reduce sync traffic happens here
                optimizer.step()
                total += n
            event_log.event("ddp_round_complete", rank=rank, round=round_index, examples=total)
            logger.info(f"[rank {rank}] round {round_index} complete ({total} examples)")
    finally:
        dist.destroy_process_group()


class DDPAdapter(DistributedFrameworkAdapter):
    def run_coordinator(self, config, logger, event_log):
        _train(config, logger, event_log, rank=0)

    def run_worker(self, config, logger, event_log):
        _train(config, logger, event_log, rank=config.worker_rank)


@DISTRIBUTED_FRAMEWORKS.register("DistributedDataParallel", implemented=True, organization="PyTorch")
def build_ddp_adapter(**kwargs):
    return DDPAdapter()
