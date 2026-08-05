"""FedLab federated learning adapter -- written to the real API; execution
hangs partway through on this machine, root cause not fully isolated.

Unlike Flower (which manages its own gRPC transport internally), FedLab's
communication layer is built directly on `torch.distributed` -- its
`DistNetwork(address, world_size, rank)` is a thin wrapper around
`dist.init_process_group(backend="gloo", init_method=f"tcp://{host}:{port}", ...)`,
the exact same TCP rendezvous mechanism this project's own
distributed_frameworks/ddp_adapter.py already uses. Server is rank 0
(`SynchronousServerManager` + `SyncServerHandler`, FedAvg aggregation over
`sample_ratio` of clients per round), each client is rank 1..N
(`PassiveClientManager` + `SGDClientTrainer`).

Reuses the same PyTorch/ResNet18/CIFAR10 combo as fl_frameworks/flower_adapter.py,
partitioned across clients the same way (`_ClientPartitionedCIFAR10` here
mirrors `_partition_loader` there) -- FedLab ships its own disk-persisting
`fedlab.contrib.dataset.PartitionedCIFAR10`, but a small in-memory wrapper
matching this project's existing dataset/family conventions is more
consistent than adopting FedLab's separate on-disk partition cache.

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

torch/torchvision/fedlab are imported lazily inside functions, not at
module scope, so `python main.py --list` can enumerate this registration
without any of them installed -- only actually running the FL slice needs
them.
"""
from core.registry import ARCHITECTURES, FL_FRAMEWORKS, FRAMEWORKS
from fl_frameworks.base import FLFrameworkAdapter


def _build_model(config):
    framework = FRAMEWORKS.get(config.framework).build()
    architecture_entry = ARCHITECTURES.get(config.architecture)
    return framework.load_model(architecture_entry, config)


class _ClientPartitionedCIFAR10:
    """Matches the `dataset.get_dataloader(id, batch_size)` interface
    fedlab.contrib.algorithm.basic_client.SGDClientTrainer.local_process
    expects -- see fedlab/contrib/algorithm/basic_client.py."""

    def __init__(self, num_clients, num_requests):
        self.num_clients = max(num_clients, 1)
        self.num_requests = num_requests

    def get_dataloader(self, client_id, batch_size):
        import numpy as np
        import torch
        import torchvision
        from torch.utils.data import DataLoader, Subset

        from families.cnn import normalize_chw

        dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        indices = list(range(client_id, len(dataset), self.num_clients))[: self.num_requests]
        subset = Subset(dataset, indices)

        def collate(batch):
            images = torch.stack([normalize_chw(np.array(img)) for img, _ in batch])
            labels = torch.tensor([label for _, label in batch])
            return images, labels

        return DataLoader(subset, batch_size=max(batch_size, 1), collate_fn=collate)


class FedLabAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        from fedlab.contrib.algorithm.basic_server import SyncServerHandler
        from fedlab.core.network import DistNetwork
        from fedlab.core.server.manager import SynchronousServerManager

        model = _build_model(config)
        handler = SyncServerHandler(model, global_round=config.num_rounds, sample_ratio=1.0)
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
        from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
        from fedlab.core.client.manager import PassiveClientManager
        from fedlab.core.network import DistNetwork

        model = _build_model(config)
        trainer = SGDClientTrainer(model)
        trainer.setup_dataset(
            _ClientPartitionedCIFAR10(config.num_clients, config.num_requests)
        )
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
