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
core/config.py's FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES and
core/training_data.py for exactly which architecture/application/dataset
combinations that supports and why (Image Classification, Sentiment
Analysis, and Activity Recognition; not e.g. Node Classification's
transductive GCN or any generative application).

torch/torchvision/flwr are imported lazily inside functions, not at module
scope, so `python main.py --list` can enumerate this registration without
any of them installed -- only actually running the FL slice needs them.
"""
from core.classification_metrics import compute_classification_metrics
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

    # BatchNorm2d (ResNet18/50, MobileNetV2) can't compute batch statistics
    # from a single sample in model.train() mode -- confirmed directly, a
    # real `ValueError: Expected more than 1 value per channel` crash with
    # this project's default config.batch_size=1. Floor at 2 and drop any
    # leftover partial batch of 1, rather than let the last batch crash.
    return DataLoader(subset, batch_size=max(config.batch_size, 2), drop_last=True)


class FlowerAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        import flwr as fl

        model = _build_model(config)

        def fit_config(server_round):
            event_log.event("fl_round_start", round=server_round)
            logger.info(f"Starting FL round {server_round}")
            # Passed through Flower's own FitIns.config to each client's
            # fit(parameters, ins_config) -- see run_client below -- so
            # every per-client phase event this round can be tagged with
            # the same round number the server logged, without the client
            # needing to track it independently.
            return {"server_round": server_round}

        def evaluate_config(server_round):
            # Same round-tagging as fit_config, for the separate evaluate()
            # RPC Flower issues after each round's fit/aggregate step.
            return {"server_round": server_round}

        def aggregate_evaluate_metrics(results):
            # results: list of (num_examples, metrics_dict) from every
            # client's evaluate() return -- see run_client below. Weighted
            # by num_examples so a client with a larger local partition
            # counts proportionally more, standard FedAvg-style weighting.
            total = sum(n for n, _ in results)
            if total == 0:
                return {}
            aggregated = {}
            for key in ("accuracy", "precision_score", "recall", "f1_score"):
                aggregated[key] = sum(n * m.get(key, 0.0) for n, m in results) / total
            event_log.event("fl_round_evaluate_aggregated", **aggregated)
            return aggregated

        strategy = fl.server.strategy.FedAvg(
            min_available_clients=config.num_clients,
            min_fit_clients=config.num_clients,
            # FedAvg's own default (2) ignores config.num_clients entirely
            # -- confirmed directly: with a genuine single-client config,
            # the server logged "configure_evaluate: no clients selected,
            # skipping evaluation" every round, silently skipping the
            # evaluate() phase (and this adapter's per-round metrics with
            # it) rather than erroring. Matches min_fit_clients above so
            # evaluate actually runs at whatever client count this
            # experiment is configured for, not just >=2.
            min_evaluate_clients=config.num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
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

        # Anchors round 1's download-phase window in telemetry/
        # round_phase_features.py's post-hoc phase-window builder --
        # otherwise there's no baseline timestamp to measure the very
        # first download against. FedLab's own adapter logs the same
        # event for the same reason.
        event_log.event("fl_client_start", client_index=client_index)

        class TorchFlowerClient(fl.client.NumPyClient):
            def get_parameters(self, ins_config):
                return _get_parameters(model)

            def fit(self, parameters, ins_config):
                server_round = ins_config.get("server_round")
                _set_parameters(model, parameters)
                # Global weights just arrived over Flower's own gRPC
                # transport and were loaded above -- this is the closest
                # this adapter code can observe of the real download,
                # since Flower's ClientApp deserializes the wire message
                # before ever calling fit().
                event_log.event("fl_download_complete", client_index=client_index, round=server_round, phase="fit")
                model.train()
                event_log.event("fl_fit_start", client_index=client_index, round=server_round)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                loss_fn = torch.nn.CrossEntropyLoss()
                total = 0
                total_loss = 0.0
                # frameworks/pytorch_adapter.py auto-places the model on
                # cuda when available, but this loader's tensors are plain
                # CPU tensors -- confirmed directly, a real "Expected all
                # tensors to be on the same device" crash on a machine that
                # actually has a GPU (every earlier test in this project
                # ran CPU-only, so this never surfaced before).
                device = next(model.parameters()).device
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(images)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    total += len(labels)
                    total_loss += loss.item() * len(labels)
                avg_loss = total_loss / total if total else 0.0
                event_log.event(
                    "fl_fit_end", client_index=client_index, round=server_round, examples=total, loss=avg_loss
                )
                event_log.event("fl_client_fit", client_index=client_index, examples=total)
                logger.info(f"Client {client_index} trained on {total} examples")
                model.eval()
                # Flower serializes and sends these returned parameters
                # over gRPC immediately after this call returns -- that
                # send happens inside Flower's own ClientApp, outside this
                # adapter's visibility, so "ready" rather than "complete"
                # is what this event actually marks.
                event_log.event("fl_upload_ready", client_index=client_index, round=server_round)
                return _get_parameters(model), max(total, 1), {}

            def evaluate(self, parameters, ins_config):
                server_round = ins_config.get("server_round")
                _set_parameters(model, parameters)
                event_log.event(
                    "fl_download_complete", client_index=client_index, round=server_round, phase="evaluate"
                )
                model.eval()
                event_log.event("fl_evaluate_start", client_index=client_index, round=server_round)
                loss_fn = torch.nn.CrossEntropyLoss()
                device = next(model.parameters()).device
                total_loss = 0.0
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        total_loss += loss_fn(output, labels).item() * len(labels)
                        all_predictions.append(output.argmax(dim=1).cpu())
                        all_labels.append(labels.cpu())
                predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
                true_labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
                num_examples = len(true_labels)
                avg_loss = (total_loss / num_examples) if num_examples else 0.0
                # Evaluated on this client's own local partition -- the
                # same one fit() just trained on, not a separate held-out
                # split (none exists in this pipeline) -- consistent with
                # this project's "traffic over accuracy" priority
                # documented elsewhere (e.g.
                # fl_frameworks/fedscale_adapter.py): these numbers track
                # convergence trend across rounds, not generalization.
                num_classes = ARCHITECTURES.get(config.architecture).meta.get("num_classes")
                metrics = compute_classification_metrics(predictions, true_labels, num_classes)
                event_log.event(
                    "fl_evaluate_end", client_index=client_index, round=server_round, loss=avg_loss, **metrics
                )
                return avg_loss, max(num_examples, 1), metrics

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
