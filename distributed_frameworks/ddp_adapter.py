"""PyTorch DistributedDataParallel adapter -- the one fully-implemented
distributed-training framework this pass. No extra dependency: DDP ships
inside `torch` already. Trains the same ResNet18/CIFAR10 combo across
processes using the gloo backend (CPU-friendly, works the same on every
device class in scope) with TCP rendezvous -- that rendezvous plus the
per-step gradient all-reduce traffic is exactly what gets captured.

torch/torchvision are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
either installed -- only actually running the DDP slice needs them.
"""
import os

from core.registry import ARCHITECTURES, DISTRIBUTED_FRAMEWORKS, FRAMEWORKS
from distributed_frameworks.base import DistributedFrameworkAdapter


def _init_process_group(config, rank):
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", config.host)
    os.environ.setdefault("MASTER_PORT", str(config.port))
    dist.init_process_group(backend="gloo", rank=rank, world_size=max(config.num_clients, 1))


def _build_loader(config):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torchvision
    from torch.utils.data import DataLoader, DistributedSampler

    from families.cnn import normalize_chw

    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    sampler = DistributedSampler(dataset, num_replicas=max(config.num_clients, 1), rank=dist.get_rank())

    def collate(batch):
        images = torch.stack([normalize_chw(np.array(img)) for img, _ in batch])
        labels = torch.tensor([label for _, label in batch])
        return images, labels

    return DataLoader(dataset, batch_size=max(config.batch_size, 1), sampler=sampler, collate_fn=collate)


def _train(config, logger, event_log, rank):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    _init_process_group(config, rank)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework)
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
