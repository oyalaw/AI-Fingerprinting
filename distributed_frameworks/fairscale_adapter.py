"""FairScale distributed-training adapter -- real implementation.

Unlike Horovod/DeepSpeed (which bring their own launcher/communication
stack), FairScale is a pure PyTorch extension library: its
`ShardedDataParallel` + `OSS` (Optimizer State Sharding, the ZeRO stage 1
technique) both take a `process_group`/`group` argument and run their
all-gather/reduce-scatter traffic over an already-initialized
`torch.distributed` process group -- the exact same gloo/TCP rendezvous
mechanism distributed_frameworks/ddp_adapter.py already sets up, confirmed
directly from `OSS.__init__`'s and `ShardedDataParallel.__init__`'s real
signatures. So this adapter reuses ddp_adapter.py's rendezvous/data-loader
setup verbatim and only swaps the model-wrapping step: `OSS(params,
optim=torch.optim.SGD, lr=0.01)` as the optimizer (a drop-in
`torch.optim.Optimizer` subclass -- `.zero_grad()`/`.step()` work
normally) wrapped in `ShardedDataParallel(model, sharded_optimizer=optimizer)`.

Same ResNet18/CIFAR10 combo as ddp_adapter.py. torch/torchvision/fairscale
are imported lazily inside functions, not at module scope, so `python
main.py --list` can enumerate this registration without any of them
installed -- only actually running this slice needs them.

**Verified**: the actually-new code here -- OSS + ShardedDataParallel
wrapping a real ResNet18 and running real forward/backward/step cycles
over a live two-process gloo/TCP process group -- was run end-to-end via
an isolated two-rank script with synthetic data (both ranks completed 3
training steps cleanly, no hang, no error). The full adapter through
main.py wasn't run start-to-finish: CIFAR10's ~170MB train split downloads
at ~70KB/s in this environment (~40 minutes), independently in both
processes, which is a pre-existing property of ddp_adapter.py's
`_build_loader` (copied here unchanged) rather than anything new to
FairScale -- the code path this adapter actually adds was the one
directly verified.
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
    from fairscale.nn.data_parallel import ShardedDataParallel
    from fairscale.optim.oss import OSS

    _init_process_group(config, rank)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework)

        optimizer = OSS(model.parameters(), optim=torch.optim.SGD, lr=0.01)
        sharded_model = ShardedDataParallel(model, sharded_optimizer=optimizer)

        loader = _build_loader(config)
        loss_fn = torch.nn.CrossEntropyLoss()

        sharded_model.train()
        for round_index in range(config.num_rounds):
            total = 0
            for images, labels in loader:
                if total >= config.num_requests:
                    break
                optimizer.zero_grad()
                output = sharded_model(images)
                loss = loss_fn(output, labels)
                loss.backward()  # reduce-scatter of sharded gradients happens here
                optimizer.step()  # OSS all-gathers updated shards back to all ranks here
                total += len(labels)
            event_log.event("fairscale_round_complete", rank=rank, round=round_index, examples=total)
            logger.info(f"[rank {rank}] round {round_index} complete ({total} examples)")
    finally:
        dist.destroy_process_group()


class FairScaleAdapter(DistributedFrameworkAdapter):
    def run_coordinator(self, config, logger, event_log):
        _train(config, logger, event_log, rank=0)

    def run_worker(self, config, logger, event_log):
        _train(config, logger, event_log, rank=config.worker_rank)


@DISTRIBUTED_FRAMEWORKS.register("FairScale", implemented=True, organization="Meta")
def build_fairscale_adapter(**kwargs):
    return FairScaleAdapter()
