"""Flower federated learning adapter -- the one fully-implemented FL
framework this pass. Wraps the same PyTorch/ResNet18/CIFAR10 combo already
used by the inference slice: each simulated client trains locally on its own
CIFAR10 partition for one local pass per round and uploads weight updates to
the Flower server, which averages them (FedAvg) and broadcasts the new
global model. Flower manages its own client/server gRPC transport
internally -- our job is just to start it with the right config and let
scapy capture whatever port it opens.

torch/torchvision/flwr are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
any of them installed -- only actually running the FL slice needs them.
"""
from core.registry import ARCHITECTURES, FL_FRAMEWORKS, FRAMEWORKS
from fl_frameworks.base import FLFrameworkAdapter


def _build_model(config):
    framework = FRAMEWORKS.get(config.framework).build()
    architecture_entry = ARCHITECTURES.get(config.architecture)
    return framework.load_model(architecture_entry, config)


def _get_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def _set_parameters(model, parameters):
    import torch

    state_dict = model.state_dict()
    for key, array in zip(state_dict.keys(), parameters):
        state_dict[key] = torch.tensor(array)
    model.load_state_dict(state_dict, strict=True)


def _partition_loader(config, client_index):
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset

    from families.cnn import normalize_chw

    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    num_clients = max(config.num_clients, 1)
    indices = list(range(client_index, len(dataset), num_clients))[: config.num_requests]
    subset = Subset(dataset, indices)

    def collate(batch):
        images = torch.stack([normalize_chw(np.array(img)) for img, _ in batch])
        labels = torch.tensor([label for _, label in batch])
        return images, labels

    return DataLoader(subset, batch_size=max(config.batch_size, 1), collate_fn=collate)


class FlowerAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        import flwr as fl

        model = _build_model(config)

        def fit_config(server_round):
            event_log.event("fl_round_start", round=server_round)
            logger.info(f"Starting FL round {server_round}")
            return {}

        strategy = fl.server.strategy.FedAvg(
            min_available_clients=config.num_clients,
            min_fit_clients=config.num_clients,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(_get_parameters(model)),
        )
        fl.server.start_server(
            server_address=f"{config.host}:{config.port}",
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
        )
        event_log.event("fl_server_complete", rounds=config.num_rounds)

    def run_client(self, config, logger, event_log):
        import flwr as fl
        import torch

        model = _build_model(config)
        client_index = config.client_index
        loader = _partition_loader(config, client_index)

        class TorchFlowerClient(fl.client.NumPyClient):
            def get_parameters(self, ins_config):
                return _get_parameters(model)

            def fit(self, parameters, ins_config):
                _set_parameters(model, parameters)
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                loss_fn = torch.nn.CrossEntropyLoss()
                total = 0
                for images, labels in loader:
                    optimizer.zero_grad()
                    output = model(images)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    total += len(labels)
                event_log.event("fl_client_fit", client_index=client_index, examples=total)
                logger.info(f"Client {client_index} trained on {total} examples")
                model.eval()
                return _get_parameters(model), max(total, 1), {}

            def evaluate(self, parameters, ins_config):
                _set_parameters(model, parameters)
                return 0.0, max(len(loader.dataset), 1), {}

        client_instance = TorchFlowerClient()
        server_address = f"{config.host}:{config.port}"
        to_client = getattr(client_instance, "to_client", None)
        if to_client is not None:
            fl.client.start_client(server_address=server_address, client=to_client())
        else:
            fl.client.start_numpy_client(server_address=server_address, client=client_instance)
        event_log.event("fl_client_complete", client_index=client_index)


@FL_FRAMEWORKS.register("Flower", implemented=True, organization="Adap GmbH")
def build_flower_adapter(**kwargs):
    return FlowerAdapter()
