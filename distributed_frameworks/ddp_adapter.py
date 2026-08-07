"""PyTorch DistributedDataParallel adapter -- the one fully-implemented
distributed-training framework this pass. No extra dependency: DDP ships
inside `torch` already. Trains the same PyTorch/Image Classification combo
across processes using the gloo backend (CPU-friendly, works the same on
every device class in scope) with TCP rendezvous -- that rendezvous plus
the per-step gradient all-reduce traffic is exactly what gets captured.

Data loading goes through core/training_data.py's
build_classification_dataset() (config.dataset, via the DATASETS/
APPLICATIONS registries) rather than hardcoding
torchvision.datasets.CIFAR10 the way this adapter originally did -- see
core/config.py's FL_DISTRIBUTED_COMPATIBLE_DATASETS for exactly which
datasets that supports and why.

torch/torchvision are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
either installed -- only actually running the DDP slice needs them.
"""
import os

from core.registry import ARCHITECTURES, DISTRIBUTED_FRAMEWORKS, FRAMEWORKS
from core.training_data import build_classification_dataset
from distributed_frameworks.base import DistributedFrameworkAdapter


def _init_process_group(config, rank):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", config.host)
    os.environ.setdefault("MASTER_PORT", str(config.port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=max(config.num_clients, 1))


def _build_loader(config):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = build_classification_dataset(config, max(config.num_clients, 1) * config.num_requests)
    sampler = DistributedSampler(dataset, num_replicas=max(config.num_clients, 1), rank=dist.get_rank())

    return DataLoader(dataset, batch_size=max(config.batch_size, 1), sampler=sampler)


def _train(config, logger, event_log, rank):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    _init_process_group(config, rank)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework, config)
        ddp_model = DistributedDataParallel(model)

        loader = _build_loader(config)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        ddp_model.train()
        for round_index in range(config.num_rounds):
            total = 0
            for images, labels in loader:
                if total >= config.num_requests:
                    break
                optimizer.zero_grad()
                output = ddp_model(images)
                loss = loss_fn(output, labels)
                loss.backward()  # gradient all-reduce sync traffic happens here
                optimizer.step()
                total += len(labels)
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
