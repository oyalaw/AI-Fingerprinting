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
from core.training_data import build_training_dataset
from core.training_objectives import get_trainable_module, get_training_step, set_trainable_module
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

    dataset = build_training_dataset(config, world_size * config.num_requests)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=dist.get_rank())

    # Same real BatchNorm2d/batch-size-1 crash documented in
    # fl_frameworks/flower_adapter.py's _partition_loader -- floor at 2,
    # drop any leftover partial batch of 1.
    return DataLoader(dataset, batch_size=max(config.batch_size, 2), sampler=sampler, drop_last=True)


def _train(config, logger, event_log, rank):
    import torch.distributed as dist
    import deepspeed

    world_size = max(config.num_clients, 1)
    _init_process_group(config, rank, world_size)
    try:
        framework = FRAMEWORKS.get(config.framework).build()
        architecture_entry = ARCHITECTURES.get(config.architecture)
        model = architecture_entry.build(framework, config)

        # architectures/ddpm.py's DDPM needs only its inner noise_predictor
        # handed to deepspeed.initialize(), not the whole T-step sampling
        # loop -- see core/training_objectives.py's get_trainable_module()
        # docstring.
        trainable = get_trainable_module(model, config.architecture)

        micro_batch = max(config.batch_size, 1)
        ds_config = {
            "train_batch_size": micro_batch * world_size,
            "train_micro_batch_size_per_gpu": micro_batch,
            "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
            "zero_optimization": {"stage": 1},
            "zero_allow_untested_optimizer": True,
        }
        model_engine, _, _, _ = deepspeed.initialize(
            model=trainable, model_parameters=trainable.parameters(), config=ds_config
        )
        model = set_trainable_module(model, config.architecture, model_engine)
        # get_trainable_module() re-derives the same model_engine reference
        # from `model` (== model_engine itself for classification/
        # reconstruction, == model.noise_predictor for denoising) -- this
        # is DeepSpeed's own engine object, the thing .backward()/.step()
        # actually get called on below, which differs from `model` in the
        # denoising case.
        engine = get_trainable_module(model, config.architecture)

        loader = _build_loader(config, world_size)
        training_step = get_training_step(config.architecture)

        model.train()
        # Same real device-mismatch crash documented in
        # fl_frameworks/flower_adapter.py's fit() -- the model can be
        # auto-placed on cuda, but this loader's tensors are plain CPU.
        # engine.device is DeepSpeed's own documented way to get the
        # engine's actual device for exactly this purpose.
        for round_index in range(config.num_rounds):
            total = 0
            for batch in loader:
                if total >= config.num_requests:
                    break
                loss, n = training_step(model, batch, engine.device)
                engine.backward(loss)  # ZeRO stage 1 gradient reduce-scatter happens here
                engine.step()  # ZeRO stage 1 optimizer-state-sharded update happens here
                total += n
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
