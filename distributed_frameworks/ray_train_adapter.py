"""Ray Train -- real implementation, verified end-to-end.

Checked directly: `pip install ray` (and `ray[train]`) both fail with
`No matching distribution found` -- Ray has no published wheel at all for
this project's Python version in this environment. Nothing to build
against; revisit once Ray ships a compatible wheel (`pip index versions
ray` is the quick way to check).

Re-checked on Ubuntu: resolved. Ray 2.57.0 ships a real `cp313`
`manylinux2014_x86_64` wheel -- confirmed directly, `pip install
ray[train]` and `import ray.train` both work cleanly (also pulls a
current `pyarrow` 25.0.1, itself now shipping a real `cp313` wheel, unlike
the old `pyarrow==17.0.0` pin fl_frameworks/pysyft_adapter.py is still
blocked on). See fl_frameworks/fedgraph_adapter.py's docstring for the
same Ray-wheel finding from the FL side.

Unlike every other adapter in this file (which all layer directly on
`torch.distributed`, coordinated by this project's own two-role
`--role server`/`--role client` process split), Ray Train's `TorchTrainer`
is a single-entry-point API: one process calls `trainer.fit()`, and Ray
itself spawns and manages `ScalingConfig(num_workers=N)` worker *processes*
internally (confirmed directly -- `ps` during a real run shows genuinely
separate `RayTrainWorker` OS processes, not threads), each running
`torch.distributed` under the hood (confirmed via Ray's own log line,
`Setting up process group for: env://`) and wrapping the model in real
`DistributedDataParallel`. That's the same shape of mismatch
fl_frameworks/nvflare_adapter.py's `simulator_run()` has with this
project's role split -- so `run_worker` isn't meaningful here either; see
its own clear error below.

**Design notes, confirmed directly by testing each one:**

- `train_loop_per_worker` runs in a fresh Ray worker process, not this
  project's own process -- confirmed this project's own modules
  (`core.registry`, `core.training_data`) import and work correctly
  inside it, but only after calling `core.registry.discover_all()` again
  inside the closure (registry state built in the coordinator process
  doesn't carry over; the same "does the subprocess see this project's
  modules" question fl_frameworks/nvflare_adapter.py's docstring flags as
  unverified for `ScriptRunner` -- here it's confirmed to just work,
  same conda venv and `sys.path`).
- `ray.train.report(...)` is a *synchronization barrier*, not a fire-and-
  forget log call -- confirmed directly: having only rank 0 call it while
  other ranks didn't produced a real deadlock (`ray.train.report` has not
  been called by all N workers in the group`, hanging indefinitely).
  Every worker calls it once per round here, for exactly that reason.
- `self.logger` (a `logging.Logger` with real file/stream handlers) is
  deliberately NOT captured inside `train_loop_per_worker` -- handlers
  wrapping open file descriptors don't survive being pickled into a
  different process meaningfully. `self.event_log` (a thin wrapper around
  a `pathlib.Path`, reopened fresh on every `.event()` call -- see
  telemetry/experiment_log.py) is safe and is what each worker uses to
  record its own per-round completion event; the coordinator's own
  `logger` is only ever used in the coordinator's own process, before and
  after `trainer.fit()`.

**Verified end-to-end**: a hand-written toy model first (two real worker
processes, confirmed distinct PIDs, real gradient steps, real losses
logged per worker per step, no hang, no error), then this project's real
ResNet18 via the same `core.registry`/`core.training_data` path every
other FL/distributed adapter uses (confirmed: both workers built a real
model, wrapped it in real DDP, ran a real forward/backward/step over real
`Synthetic` data, and reported real per-worker metrics -- `result.error`
was `None`).
"""
from core.registry import ARCHITECTURES, DISTRIBUTED_FRAMEWORKS, FRAMEWORKS
from distributed_frameworks.base import DistributedFrameworkAdapter


def _train_loop_per_worker(config, event_log):
    import torch
    from ray.train import get_context, report
    from ray.train.torch import prepare_model
    from torch.utils.data import DataLoader, DistributedSampler

    from core.registry import discover_all
    from core.training_data import build_classification_dataset

    discover_all()

    ctx = get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    framework = FRAMEWORKS.get(config.framework).build()
    architecture_entry = ARCHITECTURES.get(config.architecture)
    model = framework.load_model(architecture_entry, config)
    model = prepare_model(model)

    dataset = build_classification_dataset(config, world_size * config.num_requests)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # Same real BatchNorm2d/batch-size-1 crash documented in
    # fl_frameworks/flower_adapter.py's _partition_loader -- floor at 2,
    # drop any leftover partial batch of 1.
    loader = DataLoader(dataset, batch_size=max(config.batch_size, 2), sampler=sampler, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for round_index in range(config.num_rounds):
        total = 0
        for images, labels in loader:
            if total >= config.num_requests:
                break
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()  # DistributedDataParallel's gradient all-reduce happens here
            optimizer.step()
            total += len(labels)
        event_log.event("ray_train_round_complete", rank=rank, round=round_index, examples=total)
        # Every worker must call report() each round -- it's a synchronization
        # barrier across the whole worker group, not a fire-and-forget log call
        # (confirmed directly: skipping it on non-zero ranks hangs the group).
        report({"rank": rank, "round": round_index, "examples": total})


class RayTrainAdapter(DistributedFrameworkAdapter):
    def run_coordinator(self, config, logger, event_log):
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer

        from telemetry.experiment_log import ExperimentLog

        # Ray's worker processes run with a different cwd than this project's own
        # process -- confirmed directly: the relative events.jsonl path this
        # project's other adapters rely on (same process/cwd everywhere else)
        # resolves to a FileNotFoundError inside a Ray worker. Resolve to an
        # absolute path before it crosses the process boundary.
        worker_event_log = ExperimentLog(event_log.path.resolve())

        world_size = max(config.num_clients, 1)
        trainer = TorchTrainer(
            lambda train_loop_config: _train_loop_per_worker(config, worker_event_log),
            scaling_config=ScalingConfig(num_workers=world_size, use_gpu=False),
        )

        event_log.event("ray_train_start", num_workers=world_size, rounds=config.num_rounds)
        logger.info(f"Ray Train starting: {world_size} workers, {config.num_rounds} rounds")
        result = trainer.fit()
        if result.error is not None:
            raise result.error
        event_log.event("ray_train_complete", rounds=config.num_rounds)
        logger.info("Ray Train complete")

    def run_worker(self, config, logger, event_log):
        raise RuntimeError(
            "Ray Train's TorchTrainer manages its own worker processes internally via "
            "ScalingConfig(num_workers=...) -- there is no separate client process to "
            "launch. Run this adapter with --role server only. See "
            "distributed_frameworks/ray_train_adapter.py's module docstring for why."
        )


@DISTRIBUTED_FRAMEWORKS.register("Ray Train", implemented=True, organization="Anyscale")
def build_ray_train_adapter(**kwargs):
    return RayTrainAdapter()
