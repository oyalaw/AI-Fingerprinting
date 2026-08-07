"""Flower federated learning adapter -- the one fully-implemented FL
framework this pass. Wraps the same PyTorch/Image Classification combo
already used by the inference slice: each simulated client trains locally
on its own partition of config.dataset for one local pass per round and
uploads weight updates to the Flower server, which averages them (FedAvg)
and broadcasts the new global model. Flower manages its own client/server
gRPC transport internally -- our job is just to start it with the right
config and let scapy capture whatever port it opens.

Data loading goes through core/training_data.py's
build_classification_dataset() -- the same DATASETS/APPLICATIONS registry
abstraction paradigm=inference uses -- rather than hardcoding
torchvision.datasets.CIFAR10 the way this adapter originally did. See
core/config.py's FL_DISTRIBUTED_COMPATIBLE_DATASETS for exactly which
datasets that supports and why (CIFAR10 or Synthetic; both real 10-label
classification data, unlike e.g. ImageNet's 1000 classes).

torch/torchvision/flwr are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
any of them installed -- only actually running the FL slice needs them.
"""
from core.registry import ARCHITECTURES, FL_FRAMEWORKS, FRAMEWORKS
from core.training_data import build_classification_dataset
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
    from torch.utils.data import DataLoader, Subset

    num_clients = max(config.num_clients, 1)
    full_dataset = build_classification_dataset(config, num_clients * config.num_requests)
    indices = list(range(client_index, len(full_dataset), num_clients))[: config.num_requests]
    subset = Subset(full_dataset, indices)

    return DataLoader(subset, batch_size=max(config.batch_size, 1))


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
