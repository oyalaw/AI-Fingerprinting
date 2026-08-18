"""FedScale -- real implementation, verified end-to-end. The earlier
scoping pass (see this project's git history) concluded FedScale's real
`Aggregator(args)`/`Executor(args)` ~90-field config surface -- its own
device-simulation/executor-launcher platform, not a library a thin
adapter wraps -- made this a genuinely new sub-project. Re-investigated
by actually driving `Aggregator`/`Executor` directly with real (mostly
default) args instead of stopping at reading the config dump, the same
"confirmed by actually running it" standard the rest of this project
holds -- and it turned out tractable after all, once one real, subtle bug
was found and worked around.

**The ~90-field surface is mostly FedScale's own real defaults, not
something this adapter has to construct from scratch**: `fedscale.cloud.
config_parser.args` ships a populated, reasonable default for nearly
everything (confirmed directly by loading it) -- only a handful of fields
actually need overriding for a local CPU run: `ps_ip`/`ps_port` (this
project's own `config.host`/`config.port`), `this_rank` (0 for the
aggregator, 1 for the executor), `executor_configs` (`"{host}:[1]"` --
confirmed directly this core-count is how many concurrent registration
handshakes the aggregator waits for from that address, *not* how many
clients get simulated there; scaling it with `num_clients` made the
aggregator hang forever waiting for a second handshake this adapter's one
real Executor process never sends -- see `_build_args`'s own comment),
`num_participants` (this one *does* control how many clients get
simulated, spread across whatever executor(s) actually register),
`use_cuda=False`, and `wandb_token=""` (FedScale's own optional logging,
not required despite the default args implying otherwise).

**Aggregator/Executor genuinely map onto this project's server/client
roles**, unlike every other FL adapter in this file that needed a
workaround (NVFlare's `simulator_run()`, FedGraph's `run_GC()`): both
have a real `.run()` method (confirmed directly from the package's own
`if __name__ == "__main__"` blocks in `aggregator.py`/`executor.py`), and
running them as two real separate OS processes -- `Aggregator(args).run()`
opens a real gRPC server on `ps_port` and blocks monitoring events;
`Executor(args).run()` downloads and partitions real CIFAR10 data (via
FedScale's own downloader, not this project's `datasets/cifar10.py` --
confirmed a real ~170MB `torchvision`-style download, same slow-download
characteristic this project's own README already notes for CIFAR10),
opens a real gRPC connection to the aggregator, and both proceed -- is
confirmed to work directly, unlike FedGraph's single-entry-point shape.

**One real, subtle bug found and worked around**: with `rounds=1` (this
project's own default `num_rounds`), the aggregator's
`round_completion_handler` -- confirmed directly by reading its source --
does `self.round += 1` *before* checking `if self.round >= self.args.
rounds: broadcast(SHUT_DOWN)`, so `rounds=1` shuts down immediately after
the very first (setup) call, *before* ever broadcasting a real
`client_train` event to the executor. Confirmed directly: a run with
`args.rounds = config.num_rounds` produced zero real training (aggregator
selected participants, then immediately terminated); `args.rounds =
config.num_rounds + 1` produced a real round -- confirmed via the
executor's own log, real `Test set: Average loss: 2.3074, Top-1 Accuracy:
0.0999` (real untrained-ResNet18-on-real-CIFAR10 accuracy, ~10% = chance
for 10 classes, exactly what a random-init model should score) followed
by real `Training of (CLIENT: 1) completes`/`(CLIENT: 2) completes` with
real moving-loss values, then a real aggregation round on the aggregator
side (`Succeed participants: 2, Training loss: 2.4567...`).

**No new architecture/application/dataset registry entries needed**
(unlike fl_frameworks/fedgraph_adapter.py): FedScale's own default
`model: "resnet18"`/`model_zoo: "torchcv"` and `data_set: "cifar10"` are
genuinely ResNet18 on real CIFAR10 -- a different concrete model/loader
than this project's own architectures/resnet18.py/datasets/cifar10.py
(the same "two real implementations, same architecture name" precedent
as `graph_framework: PyTorch Geometric`'s GCN or fl_frameworks/
fedgraph_adapter.py's GIN), so ground truth correctly reuses the existing
`architecture: ResNet18`/`application: Image Classification`/`dataset:
CIFAR10` entries rather than needing new ones. Since FedScale's own data
pipeline is hardcoded to CIFAR10 internally (unlike Flower/FedLab/NVFlare,
which respect `config.dataset` via this project's own loader), this
adapter validates `config.dataset == "CIFAR10"` itself and raises a clear
error otherwise, rather than silently mislabeling ground truth if someone
sets `dataset: Synthetic` -- the same discipline
core/config.py's FEDGRAPH_COMPATIBLE_ARCHITECTURES carve-out enforces for
FedGraph, just at the adapter level here since `ResNet18`/`CIFAR10` are
still genuinely valid choices for every *other* FL adapter in this file
and don't need a new registry-wide restriction.

**Verified end-to-end**: confirmed directly with real, separate
`Aggregator`/`Executor` processes on localhost -- real gRPC connection
established, real CIFAR10 downloaded and partitioned, a real training
round dispatched and completed by both simulated clients, real
aggregation logged. `local_steps=1` (traffic generation, not
training-quality tuning, the same policy every other adapter here
follows).

Only `client_index == 0` actually launches the (single) real Executor
process -- FedScale's own simulation-mode design has one executor process
simulate every client internally (confirmed directly: the aggregator
dispatched `client_train` events for both client 1 and client 2 to the
same, single connected executor rank), not one OS process per client the
way Flower/FedLab expect; `run_client` raises a clear error for any other
`client_index` rather than silently doing nothing.
"""
from fl_frameworks.base import FLFrameworkAdapter
from core.registry import FL_FRAMEWORKS

_REQUIRED_ARCHITECTURE = "ResNet18"
_REQUIRED_DATASET = "CIFAR10"


def _check_architecture_dataset(config):
    if config.architecture != _REQUIRED_ARCHITECTURE or config.dataset != _REQUIRED_DATASET:
        raise RuntimeError(
            f"fl_framework: FedScale always runs its own real {_REQUIRED_ARCHITECTURE}/"
            f"{_REQUIRED_DATASET} pipeline internally (FedScale's own model_zoo='torchcv' "
            f"model and CIFAR10 downloader, not this project's own loader) -- got "
            f"architecture={config.architecture!r}, dataset={config.dataset!r}. Set both to "
            f"{_REQUIRED_ARCHITECTURE}/{_REQUIRED_DATASET} to avoid mislabeling ground truth. "
            f"See this module's docstring."
        )


def _build_args(config, this_rank):
    from fedscale.cloud.config_parser import args

    num_clients = max(config.num_clients, 1)
    args.use_cuda = False
    args.engine = "pytorch"
    args.data_set = "cifar10"
    args.model = "resnet18"
    args.model_zoo = "torchcv"
    args.num_participants = num_clients
    # The core-count here is how many concurrent registration handshakes the
    # aggregator waits for from this executor address, NOT how many clients
    # it simulates (that's num_participants, spread across whatever executor(s)
    # register) -- confirmed directly: scaling this with num_clients made the
    # aggregator hang forever at "Received executor 1 information, 1/2",
    # waiting for a second handshake this adapter's single real Executor
    # process never sends. Stays at 1 regardless of num_clients.
    args.executor_configs = f"{config.host}:[1]"
    args.ps_ip = config.host
    args.ps_port = str(config.port)
    args.this_rank = this_rank
    # FedScale's own round_completion_handler increments self.round *before*
    # checking `>= args.rounds` -- rounds=N shuts down after N-1 real
    # dispatched rounds, confirmed directly (see this module's docstring).
    args.rounds = config.num_rounds + 1
    args.local_steps = 1
    args.wandb_token = ""
    args.log_path = f"{config.results_dir}/fedscale_logs"
    return args


class FedScaleAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        _check_architecture_dataset(config)
        from fedscale.cloud.aggregation.aggregator import Aggregator

        args = _build_args(config, this_rank=0)

        event_log.event(
            "fedscale_start", num_clients=max(config.num_clients, 1), rounds=config.num_rounds
        )
        logger.info(
            f"FedScale Aggregator starting: real gRPC server on {config.host}:{config.port}, "
            f"{config.num_rounds} rounds, real CIFAR10/ResNet18"
        )
        Aggregator(args).run()
        event_log.event("fedscale_complete", rounds=config.num_rounds)
        logger.info("FedScale Aggregator complete")

    def run_client(self, config, logger, event_log):
        _check_architecture_dataset(config)
        if config.client_index != 0:
            raise RuntimeError(
                "FedScale's simulation-mode Executor simulates every client internally "
                "within one real process -- launch only --client-index 0. See this "
                "module's docstring for why (confirmed directly: the aggregator dispatches "
                "client_train events for every simulated client to the same executor rank)."
            )
        from fedscale.cloud.execution.executor import Executor

        args = _build_args(config, this_rank=1)

        event_log.event("fedscale_executor_start", num_clients=max(config.num_clients, 1))
        logger.info(
            f"FedScale Executor starting: connecting to {config.host}:{config.port}, "
            f"simulating {max(config.num_clients, 1)} clients, real CIFAR10/ResNet18"
        )
        Executor(args).run()
        event_log.event("fedscale_executor_complete")
        logger.info("FedScale Executor complete")


@FL_FRAMEWORKS.register("FedScale", implemented=True, organization="Michigan State University")
def build_fedscale_adapter(**kwargs):
    return FedScaleAdapter()
