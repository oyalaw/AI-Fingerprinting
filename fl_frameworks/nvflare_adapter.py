"""NVFlare federated learning adapter -- written to the real API. Two
separate, serious caveats, disclosed here rather than glossed over:

1. `import nvflare` fails outright on this Windows machine. Confirmed
   directly: nvflare/__init__.py unconditionally imports FedJob, which
   transitively imports nvflare.fuel.f3.cellnet.net_agent, which does
   `import resource` -- the POSIX resource-limits module, which doesn't
   exist on Windows at all. This is a hard platform gate baked into
   NVFlare's core import chain, not something fixable in this adapter;
   NVFlare requires Linux/macOS.

2. Separately, and more fundamentally: NVFlare's `FedJob.simulator_run()`
   -- the simplest, most direct API match for "run a federated round from
   one Python call", and what this adapter uses -- runs the server and
   all simulated clients as in-process threads communicating over
   in-memory queues, not real network sockets. That means even on a
   supported OS, this configuration produces NO capturable network
   traffic at all, which is the entire point of this project. A real
   traffic-generating NVFlare deployment needs its separate POC/production
   mode (`nvflare provision` generating per-site startup kits, each site
   run as its own `start.sh` process, coordinated via NVFlare's admin
   console) -- a substantially larger scope, not implemented here. So
   unlike frameworks/executorch_adapter.py or fl_frameworks/fedlab_adapter.py
   (both real code that WOULD serve this project's purpose if only the
   environment allowed testing), this one wouldn't produce useful traffic
   even if it did run, in this simulator-mode form.

Uses NVFlare's Client API pattern (the modern, non-deprecated way to write
NVFlare training scripts): FedJob().to(FedAvg(...), "server") for the
server-side aggregation controller, and .to(ScriptRunner(script=...), site)
per client, where the script is a real standalone training script (written
to a temp file at run time, since ScriptRunner requires a script path, not
an inline callable) using nvflare.client's flare.init()/receive()/send()
around the same PyTorch/ResNet18/CIFAR10 combo as fl_frameworks/flower_adapter.py.

run_client() isn't meaningful for this adapter: in simulator mode there is
no separate client process to launch -- simulator_run() drives every
simulated client from the single run_server() call. It raises a clear
error explaining this rather than silently doing nothing.

nvflare/torch/torchvision are imported lazily so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
on a machine without them installed.
"""
import textwrap

from core.registry import ARCHITECTURES, FL_FRAMEWORKS, FRAMEWORKS
from fl_frameworks.base import FLFrameworkAdapter

_CLIENT_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset

    import nvflare.client as flare

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


    def normalize_chw(array_hwc_uint8):
        tensor = torch.from_numpy(array_hwc_uint8.copy()).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std


    def main():
        flare.init()
        model = torchvision.models.resnet18(weights=None, num_classes=10)

        info = flare.system_info()
        site_name = flare.get_site_name()
        num_clients = max(int(info.get("num_clients", 1)), 1)
        client_index = int(site_name.split("-")[-1]) - 1 if "-" in site_name else 0

        dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        indices = list(range(client_index, len(dataset), num_clients))[:NUM_REQUESTS]
        subset = Subset(dataset, indices)

        def collate(batch):
            images = torch.stack([normalize_chw(np.array(img)) for img, _ in batch])
            labels = torch.tensor([label for _, label in batch])
            return images, labels

        loader = DataLoader(subset, batch_size=BATCH_SIZE, collate_fn=collate)

        while flare.is_running():
            input_model = flare.receive()
            model.load_state_dict({k: torch.tensor(v) for k, v in input_model.params.items()})

            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = torch.nn.CrossEntropyLoss()
            total = 0
            for images, labels in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                total += len(labels)

            output_model = flare.FLModel(
                params={k: v.cpu().numpy() for k, v in model.state_dict().items()},
                params_type="FULL",
                metrics={"trained_examples": total},
            )
            flare.send(output_model)


    if __name__ == "__main__":
        main()
    """
)


def _build_model(config):
    framework = FRAMEWORKS.get(config.framework).build()
    architecture_entry = ARCHITECTURES.get(config.architecture)
    return framework.load_model(architecture_entry, config)


class NVFlareAdapter(FLFrameworkAdapter):
    def run_server(self, config, logger, event_log):
        import tempfile
        from pathlib import Path

        from nvflare.app_common.workflows.fedavg import FedAvg
        from nvflare.job_config.api import FedJob
        from nvflare.job_config.script_runner import ScriptRunner

        model = _build_model(config)

        job = FedJob(name="ai_fingerprint_nvflare", min_clients=config.num_clients)
        job.to(FedAvg(num_clients=config.num_clients, num_rounds=config.num_rounds), "server")
        job.to(model, "server")

        script_source = _CLIENT_SCRIPT.replace(
            "NUM_REQUESTS", str(config.num_requests)
        ).replace("BATCH_SIZE", str(max(config.batch_size, 1)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "client_train.py"
            script_path.write_text(script_source)

            for i in range(1, config.num_clients + 1):
                job.to(ScriptRunner(script=str(script_path), launch_external_process=False), f"site-{i}")

            event_log.event(
                "fl_server_start", num_clients=config.num_clients, rounds=config.num_rounds
            )
            logger.info(
                f"NVFlare simulator starting: {config.num_clients} clients, "
                f"{config.num_rounds} rounds (in-process, no real network traffic -- see module docstring)"
            )
            job.simulator_run(str(Path(tmp_dir) / "workspace"), n_clients=config.num_clients)

        event_log.event("fl_server_complete", rounds=config.num_rounds)

    def run_client(self, config, logger, event_log):
        raise RuntimeError(
            "NVFlare's simulator mode has no separate client process to launch -- "
            "FedJob.simulator_run() drives every simulated client in-process from "
            "the server-side call. Run this adapter with --role server only. See "
            "fl_frameworks/nvflare_adapter.py's module docstring for why, and for "
            "why simulator mode doesn't produce real network traffic even then."
        )


@FL_FRAMEWORKS.register("NVFlare", implemented=True, organization="NVIDIA")
def build_nvflare_adapter(**kwargs):
    return NVFlareAdapter()
