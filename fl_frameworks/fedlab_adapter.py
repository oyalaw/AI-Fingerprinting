"""FedLab federated learning adapter -- real implementation, verified
end-to-end. The hang described below was specific to this project's
earlier Windows dev machine and does not reproduce on Ubuntu -- re-run
directly after this project moved to Linux, confirming the "FedLab/PyTorch
version incompatibility in its lower-level send/recv wire protocol"
hypothesis was on the right track.

Unlike Flower (which manages its own gRPC transport internally), FedLab's
communication layer is built directly on `torch.distributed` -- its
`DistNetwork(address, world_size, rank)` is a thin wrapper around
`dist.init_process_group(backend="gloo", init_method=f"tcp://{host}:{port}", ...)`,
the exact same TCP rendezvous mechanism this project's own
distributed_frameworks/ddp_adapter.py already uses. Server is rank 0
(`SynchronousServerManager` + `SyncServerHandler`, FedAvg aggregation over
`sample_ratio` of clients per round), each client is rank 1..N
(`PassiveClientManager` + `SGDClientTrainer`).

Reuses the same PyTorch/Image Classification combo as
fl_frameworks/flower_adapter.py, partitioned across clients the same way
(`_ClientPartitionedDataset` here mirrors `_partition_loader` there) --
FedLab ships its own disk-persisting `fedlab.contrib.dataset.PartitionedCIFAR10`,
but reusing core/training_data.py's registry-based loader (config.dataset,
not a hardcoded CIFAR10) matches this project's existing dataset/family
conventions more consistently than adopting FedLab's separate on-disk
partition cache. See core/config.py's FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES
and core/training_data.py for exactly which architecture/application/
dataset combinations that supports and why.

**What was actually tested here, and what's still unresolved:** a real bug
was found and fixed -- `SyncServerHandler` lives in
`fedlab.contrib.algorithm.basic_server`, not `fedlab.core.server.handler`
(only the abstract `ServerHandler` base class lives there). Bare
`torch.distributed` gloo rendezvous over TCP was verified working correctly
on this Windows machine via an isolated two-process script, independent of
FedLab entirely. But running the real client/server pair through this
adapter hangs indefinitely right after connecting, with no error on either
side. Narrowed down with a further isolated test of just
`ServerManager.setup()` / `ClientManager.setup()` (FedLab's own post-connect
handshake, where each client sends one `MessageCode.SetUp` message the
server waits to receive): the client's `setup()` call completes and
returns normally, but the server's matching `setup()` call -- blocked on
`self._network.recv(src=rank)` -- never returns, even though the client's
send reported success. That's an asymmetric hang (one side's call returns,
the matching side's never does), which points at a FedLab/PyTorch version
incompatibility in its lower-level send/recv wire protocol rather than a
one-line fix, but that's a hypothesis from investigation, not a confirmed
root cause -- don't rule out a bug in this adapter's own setup either. Re-run
the isolated setup diagnostics yourself before trusting this one; treat it
like frameworks/executorch_adapter.py and frameworks/tvm_adapter.py, i.e.
real code against the documented API, not confirmed to run end-to-end.

**Update, on Ubuntu**: re-ran this exact adapter, unchanged, through
`python main.py` for real. No hang at all -- confirmed directly with both
a 1-client and a 2-client run: server and client(s) complete the full
`SetUp` -> `ParameterUpdate` -> `Exit` message sequence cleanly in each
direction (visible in FedLab's own network logging -- every `Sent`/
`Received` pair on both sides, no missing acknowledgement), both processes
exit 0, and `ground_truth.json` correctly records
`"fl_framework": "FedLab"`. The asymmetric-hang hypothesis above (a
FedLab/PyTorch version incompatibility in the send/recv wire protocol,
not a bug in this adapter's own code) is the more likely explanation in
hindsight, given the exact same adapter code now runs cleanly end-to-end
with no changes -- only the OS/environment did. Now genuinely verified,
not just written-to-the-API.

torch/torchvision/fedlab are imported lazily inside functions, not at
module scope, so `python main.py --list` can enumerate this registration
without any of them installed -- only actually running the FL slice needs
them.
"""
from core.classification_metrics import compute_classification_metrics
from core.registry import ARCHITECTURES, FL_FRAMEWORKS, FRAMEWORKS
from core.training_data import build_classification_dataset
from fl_frameworks.base import FLFrameworkAdapter


def _build_model(config):
    framework = FRAMEWORKS.get(config.framework).build()
    architecture_entry = ARCHITECTURES.get(config.architecture)
    return framework.load_model(architecture_entry, config)


class _ClientPartitionedDataset:
    """Matches the `dataset.get_dataloader(id, batch_size)` interface
    fedlab.contrib.algorithm.basic_client.SGDClientTrainer.local_process
    expects -- see fedlab/contrib/algorithm/basic_client.py."""

    def __init__(self, config):
        self.config = config
        self.num_clients = max(config.num_clients, 1)

    def get_dataloader(self, client_id, batch_size):
        from torch.utils.data import DataLoader, Subset

        full_dataset = build_classification_dataset(
            self.config, self.num_clients * self.config.num_requests
        )
        indices = list(range(client_id, len(full_dataset), self.num_clients))[
            : self.config.num_requests
        ]
        subset = Subset(full_dataset, indices)

        # Same real BatchNorm2d/batch-size-1 crash documented in
        # fl_frameworks/flower_adapter.py's _partition_loader -- floor at 2,
        # drop any leftover partial batch of 1.
        return DataLoader(subset, batch_size=max(batch_size, 2), drop_last=True)


def _round_aware_handler_cls():
    """SyncServerHandler.downlink_package only ever sends [model_parameters]
    -- confirmed directly by reading fedlab/contrib/algorithm/basic_server.py
    -- so a client's local_process()/train() has no way to know which round
    it's training, unlike Flower's on_fit_config_fn which hands the client
    the round number explicitly. FedLab's own AsyncServerHandler already
    establishes the pattern this borrows: downlink_package =
    [self.model_parameters, torch.Tensor([self.round])]. Subclassed rather
    than monkeypatched so the rest of SyncServerHandler's real, verified
    aggregation logic (see this module's own docstring) is untouched."""
    import torch
    from fedlab.contrib.algorithm.basic_server import SyncServerHandler

    class RoundAwareServerHandler(SyncServerHandler):
        @property
        def downlink_package(self):
            return [self.model_parameters, torch.Tensor([float(self.round)])]

    return RoundAwareServerHandler


def _event_logging_trainer_cls():
    """SGDClientTrainer.train() is FedLab's own documented customization
    point ("Overwrite this method to customize the PyTorch training
    pipeline" -- fedlab/contrib/algorithm/basic_client.py's own
    SGDSerialClientTrainer.train() docstring, same pattern here). Adds
    the same phase-boundary event logging + real classification metrics
    fl_frameworks/flower_adapter.py's NumPyClient.fit()/evaluate() do,
    without touching FedLab's own SGD training loop."""
    import torch
    from fedlab.contrib.algorithm.basic_client import SGDClientTrainer

    class EventLoggingClientTrainer(SGDClientTrainer):
        def configure_logging(self, config, logger, event_log, client_index):
            self._fp_config = config
            self._fp_logger = logger
            self._fp_event_log = event_log
            self._fp_client_index = client_index

        def local_process(self, payload, id):
            model_parameters = payload[0]
            # FedLab's own round counter is 0-indexed (increments only
            # after a round's aggregation completes); +1 here so
            # round=1 means "the first round" the same way Flower's own
            # server_round does, for a consistent join key across
            # fl_framework values downstream.
            fedlab_round = int(payload[1].item()) if len(payload) > 1 else None
            round_number = (fedlab_round + 1) if fedlab_round is not None else None
            train_loader = self.dataset.get_dataloader(id, self.batch_size)
            self.train(model_parameters, train_loader, round_number=round_number)

        def train(self, model_parameters, train_loader, round_number=None) -> None:
            from fedlab.utils import SerializationTool

            event_log = self._fp_event_log
            client_index = self._fp_client_index

            SerializationTool.deserialize_model(self._model, model_parameters)
            # Global weights just arrived over FedLab's own torch.distributed
            # gloo transport and were loaded above -- the closest this
            # adapter code can observe of the real download.
            event_log.event(
                "fl_download_complete", client_index=client_index, round=round_number, phase="fit"
            )
            event_log.event("fl_fit_start", client_index=client_index, round=round_number)
            self._model.train()
            total = 0
            total_loss = 0.0
            for _ in range(self.epochs):
                for data, target in train_loader:
                    if self.cuda:
                        data, target = data.cuda(self.device), target.cuda(self.device)
                    outputs = self._model(data)
                    loss = self.criterion(outputs, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total += len(target)
                    total_loss += loss.item() * len(target)
            avg_loss = total_loss / total if total else 0.0
            event_log.event(
                "fl_fit_end", client_index=client_index, round=round_number, examples=total, loss=avg_loss
            )

            # FedLab's basic FedAvg pattern (unlike Flower's FedAvg
            # strategy) has no separate server-driven evaluate RPC at
            # all -- confirmed directly, SyncServerHandler/
            # PassiveClientManager only ever exchange ParameterUpdate
            # messages. Evaluating immediately here, on the same local
            # partition training just used (no held-out split exists in
            # this pipeline), is the closest equivalent -- same "traffic
            # over accuracy" simplification fl_frameworks/flower_adapter.py
            # documents for its own evaluate().
            event_log.event("fl_evaluate_start", client_index=client_index, round=round_number)
            self._model.eval()
            all_predictions, all_labels = [], []
            eval_loss_total = 0.0
            with torch.no_grad():
                for data, target in train_loader:
                    if self.cuda:
                        data, target = data.cuda(self.device), target.cuda(self.device)
                    output = self._model(data)
                    eval_loss_total += self.criterion(output, target).item() * len(target)
                    all_predictions.append(output.argmax(dim=1).cpu())
                    all_labels.append(target.cpu())
            predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
            labels_tensor = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
            eval_examples = len(labels_tensor)
            eval_avg_loss = (eval_loss_total / eval_examples) if eval_examples else 0.0
            num_classes = ARCHITECTURES.get(self._fp_config.architecture).meta.get("num_classes")
            metrics = compute_classification_metrics(predictions, labels_tensor, num_classes)
            event_log.event(
                "fl_evaluate_end", client_index=client_index, round=round_number, loss=eval_avg_loss, **metrics
            )
            self._model.train()

            event_log.event("fl_upload_ready", client_index=client_index, round=round_number)
            self._fp_logger.info(
                f"Client {client_index} round {round_number}: trained on {total} examples, "
                f"accuracy={metrics['accuracy']:.4f}"
            )

    return EventLoggingClientTrainer


class FedLabAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        from fedlab.core.network import DistNetwork
        from fedlab.core.server.manager import SynchronousServerManager

        model = _build_model(config)
        handler = _round_aware_handler_cls()(model, global_round=config.num_rounds, sample_ratio=1.0)
        handler.num_clients = config.num_clients

        network = DistNetwork(
            address=(config.host, config.port), world_size=config.num_clients + 1, rank=0
        )
        manager = SynchronousServerManager(network=network, handler=handler, mode="LOCAL")

        event_log.event("fl_server_start", num_clients=config.num_clients, rounds=config.num_rounds)
        logger.info(f"FedLab server starting: world_size={config.num_clients + 1}, rank=0")
        manager.run()
        event_log.event("fl_server_complete", rounds=config.num_rounds)

    def run_client(self, config, logger, event_log):
        from fedlab.core.client.manager import PassiveClientManager
        from fedlab.core.network import DistNetwork

        model = _build_model(config)
        trainer = _event_logging_trainer_cls()(model)
        trainer.configure_logging(config, logger, event_log, config.client_index)
        trainer.setup_dataset(_ClientPartitionedDataset(config))
        trainer.setup_optim(epochs=1, batch_size=config.batch_size, lr=0.01)

        rank = config.client_index + 1
        network = DistNetwork(
            address=(config.host, config.port), world_size=config.num_clients + 1, rank=rank
        )
        manager = PassiveClientManager(network=network, trainer=trainer)

        event_log.event("fl_client_start", client_index=config.client_index, rank=rank)
        logger.info(f"FedLab client {config.client_index} starting: rank={rank}")
        manager.run()
        event_log.event("fl_client_complete", client_index=config.client_index)


@FL_FRAMEWORKS.register("FedLab", implemented=True, organization="SMILELab")
def build_fedlab_adapter(**kwargs):
    return FedLabAdapter()
