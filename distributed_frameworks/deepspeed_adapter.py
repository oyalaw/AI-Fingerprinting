"""DeepSpeed -- real implementation. Re-investigated on Ubuntu (the
original findings below are all from this project's earlier Windows dev
machine): the "Windows/no-CUDA op-builder" wall is fully resolved on
Linux, and DeepSpeed's ZeRO stage 1 optimizer now runs for real, verified
end-to-end over a live two-process gloo process group.

Original finding: `pip install --no-build-isolation deepspeed` failed
deep inside its own `build_ext` step with a symptom of its op-builder
enumerating zero compilable sources for a Windows/no-CUDA target --
consistent with DeepSpeed being fundamentally built around ahead-of-time
CUDA op compilation.

On Ubuntu, confirmed directly: plain `pip install deepspeed` (no compiler
flags needed at all) succeeds cleanly and quickly -- DeepSpeed's modern
default install mode doesn't ahead-of-time-compile any ops, it JIT-builds
individual ops via `ninja` the first time each one is actually used.
`import deepspeed` then prints `Setting accelerator to CPU. If you have
GPU or other accelerator, we were unable to detect it.` -- an informational
warning, not an error; DeepSpeed has a real CPU accelerator backend for
exactly this environment. Confirmed further: `deepspeed.init_distributed
(dist_backend="gloo", rank=rank, world_size=world_size)` (the framework's
own wrapper around `torch.distributed.init_process_group`, needed so its
internal comm backend picks up the right rank/world_size) JIT-builds one
real op the first time it's called (`deepspeed_shm_comm_op`, needs `ninja`
on `PATH` -- already a pip-installed dependency of this project's other
adapters), then `deepspeed.initialize(model=..., model_parameters=...,
config={"zero_optimization": {"stage": 1}, ...})` genuinely constructs a
`DeepSpeedEngine` wrapping ZeRO stage 1 (optimizer state sharding, the
same technique distributed_frameworks/fairscale_adapter.py's `OSS`
implements independently) over that process group. Needed one extra
config key past a plain SGD optimizer: DeepSpeed's own sanity check
rejects `torch.optim.SGD` as "untested" for ZeRO unless
`"zero_allow_untested_optimizer": True` is set explicitly (Adam/AdamW are
its normal ZeRO-tested optimizers) -- confirmed directly this is
DeepSpeed's own documented escape hatch for exactly this case, not a
workaround for a bug.

**Verified**: `deepspeed.initialize()` wrapping a real model and running
real `model_engine(...)`/`model_engine.backward(loss)`/`model_engine.step()`
cycles over a live two-process gloo process group was first spot-checked
via an isolated two-rank script with a toy model and synthetic tensors
(both ranks completed 3 training steps cleanly with real, distinct loss
values each step, no hang, no error) -- the same verification shape
distributed_frameworks/fairscale_adapter.py's docstring describes. Then
run for real, full end-to-end through this actual adapter via `python
main.py`: `architecture: ResNet18`/`application: Image Classification`/
`dataset: Synthetic`/`distributed_framework: DeepSpeed`, coordinator (rank
0) and worker (rank 1) as two real separate processes -- both completed 2
full rounds cleanly, and `ground_truth.json` correctly recorded
`"distributed_framework": "DeepSpeed"` for both roles.
"""
import os

from core.registry import ARCHITECTURES, DISTRIBUTED_FRAMEWORKS, FRAMEWORKS
from core.training_data import build_classification_dataset
from distributed_frameworks.base import DistributedFrameworkAdapter


def _init_process_group(config, rank, world_size):
    os.environ.setdefault("MASTER_ADDR", config.host)
    os.environ.setdefault("MASTER_PORT", str(config.port))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import deepspeed

    deepspeed.init_distributed(dist_backend="gloo", rank=rank, world_size=world_size)


def _build_loader(config, world_size):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = build_classification_dataset(config, world_size * config.num_requests)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=dist.get_rank())

    return DataLoader(dataset, batch_size=max(config.batch_size, 1), sampler=sampler)


def _train(config, logger, event_log, rank):
    import torch.distributed as dist
    import deepspeed

    world_size = max(config.num_clients, 1)
    _init_process_group(config, rank, world_size)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework, config)

        micro_batch = max(config.batch_size, 1)
        ds_config = {
            "train_batch_size": micro_batch * world_size,
            "train_micro_batch_size_per_gpu": micro_batch,
            "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
            "zero_optimization": {"stage": 1},
            "zero_allow_untested_optimizer": True,
        }
        model_engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=ds_config
        )

        loader = _build_loader(config, world_size)
        import torch

        loss_fn = torch.nn.CrossEntropyLoss()

        model_engine.train()
        for round_index in range(config.num_rounds):
            total = 0
            for images, labels in loader:
                if total >= config.num_requests:
                    break
                output = model_engine(images)
                loss = loss_fn(output, labels)
                model_engine.backward(loss)  # ZeRO stage 1 gradient reduce-scatter happens here
                model_engine.step()  # ZeRO stage 1 optimizer-state-sharded update happens here
                total += len(labels)
            event_log.event("deepspeed_round_complete", rank=rank, round=round_index, examples=total)
            logger.info(f"[rank {rank}] round {round_index} complete ({total} examples)")
    finally:
        dist.destroy_process_group()


class DeepSpeedAdapter(DistributedFrameworkAdapter):
    def run_coordinator(self, config, logger, event_log):
        _train(config, logger, event_log, rank=0)

    def run_worker(self, config, logger, event_log):
        _train(config, logger, event_log, rank=config.worker_rank)


@DISTRIBUTED_FRAMEWORKS.register("DeepSpeed", implemented=True, organization="Microsoft")
def build_deepspeed_adapter(**kwargs):
    return DeepSpeedAdapter()
