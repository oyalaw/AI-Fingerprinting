"""FedGraph -- real implementation, verified end-to-end. The scoping
investigation below concluded this needed a genuinely new sub-project, not
a thin adapter -- re-investigated further, with the scope actually broken
down by FedGraph's own task types rather than treated as one monolithic
API, and one of the three (graph classification) turned out tractable.

`fedgraph.federated_methods.run_NC`/`run_GC`/`run_LP` (node
classification / graph classification / link prediction) are genuinely
different shapes, confirmed directly rather than assumed from the
package's docs:

- `run_NC` -- confirmed to hit the exact same transductive/whole-graph
  wall this project's own architectures/gcn.py docstring already
  documents (one graph, classify every node). Ruled out for the same
  structural reason, not re-investigated further.
- `run_LP` -- link prediction, a different task shape again (predicting
  edges, not per-sample labels); not investigated, out of scope here.
- `run_GC` -- **confirmed genuinely tractable**. Graph classification is
  sample-shaped, not transductive: `fedgraph.data_process.
  data_loader_GC_single(datapath, dataset="MUTAG", num_trainer=N,
  batch_size=B)` partitions one real benchmark dataset (confirmed
  directly: real `torch_geometric.datasets.TUDataset` download, 188 real
  MUTAG molecular graphs) across `N` simulated clients -- the same shape
  every other FL adapter here partitions CIFAR10/IMDB across clients,
  just with `fedgraph`'s own loader instead of
  core/training_data.py's, and a real graph instead of an image/token
  sequence as the per-client data.

Confirmed by actually running it before writing any adapter code: a
standalone script calling `run_GC(args, data)` with `algorithm="FedAvg"`,
2 simulated trainers, real MUTAG data -- completed end-to-end. Real
separate Ray actor processes (confirmed distinct PIDs), a real GIN model
trained via `fedgraph.gnn_models.GIN` (FedGraph's own implementation, not
this project's architectures/gin.py -- see below for why both exist), and
real per-client test accuracy reported (0.6-0.7 range, consistent with
one local epoch on ~90 real training graphs per client). `run_GC`
manages its own `ray.init()` and its own `Trainer`/`Server_GC`
construction internally -- the same "single entry point, Ray manages its
own worker processes" shape as distributed_frameworks/ray_train_adapter.py,
so (like that adapter, and like fl_frameworks/nvflare_adapter.py's
simulator mode) there's no separate client role to launch here either;
see `run_client` below.

**New registry entries, not a routing trick**: FedGraph's real graph
classification task doesn't fit any existing architecture/application/
dataset combo here (the same well-justified-gap-fill precedent as
datasets/karate_club.py/architectures/gcn.py for Node Classification) --
added architectures/gin.py (a real, independently-usable
`torch_geometric`-based GIN classifier for `paradigm: inference` too, not
just this FL path), applications/graph_classification.py, and
datasets/mutag.py. `core/config.py`'s `FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES`
gate -- which locks every other FL/distributed adapter to the 8
classification architectures whose training loop this project's own
`core/training_data.py` loader supports -- doesn't apply to this adapter,
since `run_GC` never touches that loader at all; see
`FEDGRAPH_COMPATIBLE_ARCHITECTURES` in core/config.py for the narrow,
explicit carve-out (only `fl_framework: FedGraph` + `architecture: GIN`,
nothing else is affected).

Two real, distinct `GIN` implementations exist in this project now, the
same "same architecture name, different concrete class" precedent as
`graph_framework: PyTorch Geometric`'s GCN vs architectures/gcn.py's
hand-rolled one: architectures/gin.py's `_GINClassifier` (used for
`paradigm: inference`, and for every other fl_framework/distributed_framework
if one is ever wired to GIN) and `fedgraph.gnn_models.GIN` (used
internally by FedGraph's own `run_GC`, since bypassing FedGraph's real
training pipeline to force its base model to swap in a different GIN
implementation would defeat the point of testing FedGraph's own real
code path). Both are genuinely GIN; ground truth records `architecture:
GIN` correctly either way.

`num_rounds`/`num_clients`/`batch_size` come from this project's own
config, same as every other FL adapter; `algorithm` is fixed to
`"FedAvg"` (FedGraph also implements FedProx/GCFL/GCFL+/GCFL+dWs, out of
scope to expose all of here) and `local_epoch` to 1 (traffic generation,
not training-quality tuning, the same policy every other adapter here
follows).

Per-round phase/metrics logging (fl_download_complete/fl_fit_start/
fl_fit_end/fl_upload_ready/fl_evaluate_start/fl_evaluate_end, matching
fl_frameworks/flower_adapter.py's/fedlab_adapter.py's event names) is
added by _logging_trainer_gc_cls()/_logging_server_gc_cls() below --
unlike those two adapters, run_GC() builds its own Trainer_GC/Server_GC
instances internally with a fixed constructor signature, so there's no
clean instance-injection point the way FedLab/Flower have. Instead,
run_server() monkeypatches the module-level `fedgraph.federated_methods.
Trainer_GC`/`.Server_GC` names right before calling run_GC() (both are
late-bound globals `run_GC`'s own function body looks up at call time,
confirmed directly reading its source) and restores them afterward.
Every override still calls super() and never touches FedGraph's real
training/aggregation logic -- narrower than it sounds, just a different
injection mechanism than the other two adapters' clean subclass-and-pass-
an-instance pattern.

Two real, confirmed structural differences from Flower/FedLab/FedScale,
not modeling choices made here: (1) run_GC_Fed_algorithm's own FedAvg
loop only calls local_test() ONCE, after every communication round has
already finished -- not per-round -- so fl_evaluate_start/end carry no
round number. (2) neither update_params nor local_train ever receives a
round number from the driver loop, so each Trainer_GC instance
self-tracks its own round counter via an incrementing instance
attribute, incremented in update_params (called once before round 1 and
once after every round's aggregation) -- confirmed directly this
produces one extra download event (round N+1) right before the final
evaluate, which is real: the driver's own loop really does re-broadcast
the aggregated model one last time after the last round completes,
before ever calling local_test().

Verified end-to-end with a real 2-trainer run: real per-round download/
fit/upload events and real per-trainer training loss/accuracy for both
simulated clients (real separate Ray actor processes), and real final
evaluate loss/accuracy matching FedGraph's own printed test_acc values
exactly (0.6/0.7).
"""
import tempfile

from core.registry import FL_FRAMEWORKS
from fl_frameworks.base import FLFrameworkAdapter


def _logging_trainer_gc_cls(event_log):
    """Trainer_GC (fedgraph/trainer_class.py) is a plain class with no
    callback/event abstraction -- run_GC() (fedgraph/federated_methods.py)
    builds it from a dynamically-defined Ray-actor subclass
    (`class Trainer(Trainer_GC): ...`) *inside its own function body*,
    referencing the module-level `Trainer_GC` name as a late-bound global
    at call time -- confirmed directly reading its source. Monkeypatching
    `fedgraph.federated_methods.Trainer_GC` right before calling run_GC()
    (see run_server() below) makes run_GC()'s own subclass definition
    pick this one up as ITS base class instead, with zero changes to
    FedGraph's real training/aggregation logic -- every override here
    calls super().

    event_log is captured by closure, not smuggled through `args` the
    way fl_frameworks/fedlab_adapter.py's round number had to be: Ray
    actors serialize via cloudpickle, which (unlike stdlib pickle)
    correctly handles closures over plain picklable data like
    ExperimentLog (just a path + a dict, no open file handles) -- the
    same reasoning distributed_frameworks/ray_train_adapter.py's own
    docstring already establishes for its worker_event_log."""
    from fedgraph.trainer_class import Trainer_GC

    class LoggingTrainerGC(Trainer_GC):
        def _fp_event(self, event_type, **fields):
            event_log.event(event_type, client_index=self.id, **fields)

        def update_params(self, server_params):
            super().update_params(server_params)
            # run_GC_Fed_algorithm calls update_params once before round 1
            # (the initial broadcast) and once again after every round's
            # aggregation -- self-tracked here since neither update_params
            # nor local_train ever receives a round number from the
            # driver loop (confirmed directly reading federated_methods.py).
            self._fp_round = getattr(self, "_fp_round", 0) + 1
            self._fp_event("fl_download_complete", round=self._fp_round)

        def local_train(self, local_epoch, train_option="basic", mu=1):
            round_number = getattr(self, "_fp_round", 0)
            self._fp_event("fl_fit_start", round=round_number)
            super().local_train(local_epoch, train_option=train_option, mu=mu)
            training_losses = self.train_stats.get("trainingLosses") or []
            training_accs = self.train_stats.get("trainingAccs") or []
            self._fp_event(
                "fl_fit_end",
                round=round_number,
                loss=training_losses[-1] if training_losses else None,
                accuracy=training_accs[-1] if training_accs else None,
            )
            self._fp_event("fl_upload_ready", round=round_number)

        def local_test(self, test_option="basic", mu=1):
            # run_GC_Fed_algorithm only calls local_test ONCE, after every
            # communication round has already finished (confirmed
            # directly) -- unlike Flower/FedLab/FedScale's per-round
            # evaluate, so this event carries no meaningful round number.
            self._fp_event("fl_evaluate_start")
            result = super().local_test(test_option=test_option, mu=mu)
            test_loss, test_acc = result[0], result[1]
            self._fp_event("fl_evaluate_end", loss=test_loss, accuracy=test_acc)
            return result

    return LoggingTrainerGC


def _logging_server_gc_cls(event_log, logger):
    """Server_GC (fedgraph/server_class.py) runs in the driver process,
    not a Ray actor -- no serialization concerns, so this closes over
    event_log/logger directly. Same monkeypatch mechanism as
    _logging_trainer_gc_cls above: run_GC() references the module-level
    `Server_GC` name as a late-bound global when constructing it."""
    from fedgraph.server_class import Server_GC

    class LoggingServerGC(Server_GC):
        def aggregate_weights(self, selected_trainers):
            round_number = getattr(self, "_fp_round", 0) + 1
            self._fp_round = round_number
            super().aggregate_weights(selected_trainers)
            event_log.event(
                "fl_round_aggregate_complete", round=round_number, participants=len(selected_trainers)
            )
            logger.info(f"FedGraph round {round_number}: aggregated {len(selected_trainers)} trainers")

    return LoggingServerGC


class FedGraphAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        import attridict

        from fedgraph.data_process import data_loader_GC_single
        import fedgraph.federated_methods as fedgraph_federated_methods
        from fedgraph.federated_methods import run_GC

        num_trainers = max(config.num_clients, 1)

        args = attridict.AttriDict(
            {
                "algorithm": "FedAvg",
                "hidden": 16,
                "nlayer": 3,
                "dropout": 0.5,
                "device": "cpu",
                "gpu": False,
                "use_cluster": False,
                "local_epoch": 1,
                "num_rounds": config.num_rounds,
                "mu": 0.01,
                "lr": 0.001,
                "weight_decay": 0.0005,
                "save_files": False,
                "seed": 42,
                "outbase": f"{config.results_dir}/fedgraph_out",
                "num_cpus_per_trainer": 1,
                "num_gpus_per_trainer": 0,
            }
        )

        with tempfile.TemporaryDirectory() as datapath:
            data = data_loader_GC_single(
                datapath=datapath,
                dataset="MUTAG",
                num_trainer=num_trainers,
                batch_size=max(config.batch_size, 1),
            )

            event_log.event("fedgraph_start", num_trainers=num_trainers, rounds=config.num_rounds)
            logger.info(
                f"FedGraph run_GC starting: {num_trainers} simulated trainers (real Ray "
                f"actor processes), {config.num_rounds} rounds, real MUTAG graph data"
            )

            # Ray actors run in separate worker processes with a
            # different cwd than this driver process -- same real,
            # confirmed issue distributed_frameworks/ray_train_adapter.py's
            # own docstring documents for the identical reason (a
            # relative events.jsonl path resolves wrong inside a Ray
            # worker) -- resolve to an absolute path before it crosses
            # the process boundary.
            from telemetry.experiment_log import ExperimentLog

            trainer_event_log = ExperimentLog(event_log.path.resolve(), common_fields=event_log.common_fields)

            original_trainer_gc = fedgraph_federated_methods.Trainer_GC
            original_server_gc = fedgraph_federated_methods.Server_GC
            fedgraph_federated_methods.Trainer_GC = _logging_trainer_gc_cls(trainer_event_log)
            fedgraph_federated_methods.Server_GC = _logging_server_gc_cls(event_log, logger)
            try:
                run_GC(args, data)
            finally:
                # Restored unconditionally, even on failure -- these are
                # module-global attributes on the real fedgraph package,
                # not local to this call.
                fedgraph_federated_methods.Trainer_GC = original_trainer_gc
                fedgraph_federated_methods.Server_GC = original_server_gc

            event_log.event("fedgraph_complete", rounds=config.num_rounds)
            logger.info("FedGraph run_GC complete")

    def run_client(self, config, logger, event_log):
        raise RuntimeError(
            "FedGraph's run_GC() manages its own Ray-actor trainer processes internally "
            "via num_trainer= -- there is no separate client process to launch. Run this "
            "adapter with --role server only. See "
            "fl_frameworks/fedgraph_adapter.py's module docstring for why."
        )


@FL_FRAMEWORKS.register("FedGraph", implemented=True, organization="Rutgers University")
def build_fedgraph_adapter(**kwargs):
    return FedGraphAdapter()
