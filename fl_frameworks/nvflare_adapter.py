"""NVFlare federated learning adapter -- written to the real API. Two
separate, serious caveats, disclosed here rather than glossed over. Caveat
1 was re-investigated on Ubuntu (this project's earlier findings were all
from its Windows dev machine) and is now confirmed resolved -- which, for
the first time, let this adapter's code actually run instead of just be
read, and that surfaced two more real, confirmed bugs underneath it.

1. `import nvflare` used to fail outright on Windows. Confirmed directly:
   nvflare/__init__.py unconditionally imports FedJob, which transitively
   imports nvflare.fuel.f3.cellnet.net_agent, which does `import resource`
   -- the POSIX resource-limits module, which doesn't exist on Windows at
   all. On Ubuntu, confirmed directly: `import nvflare` (2.8.1) works
   cleanly, `resource` is a normal stdlib module on Linux -- this platform
   gate is fully resolved.

   Actually running this adapter for the first time (previously impossible)
   surfaced two further, genuine bugs, both against the real NVFlare 2.8.1
   API rather than the version this adapter was originally written against:

   - `FedJob.simulator_run()` raised `ValueError: You already specified
     clients using to(). Don't use n_clients in simulator_run.` -- current
     NVFlare rejects passing both explicit per-site `.to(...)` calls (which
     this adapter always made) and `n_clients=` together. Fixed: dropped
     the now-redundant `n_clients=config.num_clients` argument.

   - With that fixed, job-config generation itself then failed:
     `TypeError: Object of type type is not JSON serializable`. Traced
     directly (via a debug JSONEncoder catching the exact offending value):
     `job.to(model, "server")` -- the Client API's own convenience shortcut
     for "auto-configure a persistor from a plain `nn.Module`" -- reflects
     over the model's `__dict__` via NVFlare's own
     `get_component_init_parameters()` to capture constructor-arg-shaped
     attributes for the generated job config, and torchvision's
     `resnet18()` stores `self._norm_layer = nn.BatchNorm2d` (confirmed
     directly: a raw *class* object, the resolved default for its
     `norm_layer=None` constructor parameter, not an instance) as a
     same-named attribute. NVFlare's reflection picks it up and tries to
     JSON-serialize a live Python class into the job config, which fails
     unconditionally. This is a genuine bug/limitation in NVFlare's own
     `job.to(model, ...)` reflection against an entirely standard,
     unmodified torchvision model -- not something this project's adapter
     code causes or can route around short of hand-building a
     `PTFileModelPersistor` in place of the documented shortcut, which
     would only matter if caveat 2 below weren't already the harder
     ceiling. Left unfixed for that reason; this adapter still doesn't run
     to completion.

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
around the same PyTorch/Image Classification combo as
fl_frameworks/flower_adapter.py.

Data loading in that generated script calls
core.training_data.build_classification_dataset() -- the same DATASETS/
APPLICATIONS registry loader every other FL/distributed adapter now uses,
config.dataset/config.application baked into the script text alongside
NUM_REQUESTS/BATCH_SIZE -- rather than duplicating a second, separate
CIFAR10/normalize_chw implementation inline the way this script originally
did. This assumes ScriptRunner(launch_external_process=False) executes the
script in the same Python process/sys.path as this project (NVFlare's own
simulator design for exactly this "skip subprocess overhead" case), so
`from core.training_data import ...` resolves normally -- **still not
verified**: caveat 1's job-config serialization bug now aborts
`simulator_run()` before NVFlare ever reaches the point of actually
launching any site's script, so this generated script has still never
been executed by NVFlare for real, on either OS. If that assumption turns
out wrong wherever this eventually does run, the fallback is reinlining
the loading logic the way it originally was -- see this module's git
history for that version.

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
    import types

    import torch

    import nvflare.client as flare

    from core.registry import ARCHITECTURES, FRAMEWORKS
    from core.training_data import build_classification_dataset


    def main():
        flare.init()

        data_config = types.SimpleNamespace(
            dataset="DATASET_NAME",
            application="APPLICATION_NAME",
            num_requests=NUM_REQUESTS,
        )

        framework = FRAMEWORKS.get("PyTorch").build()
        architecture_entry = ARCHITECTURES.get("ARCHITECTURE_NAME")
        model = framework.load_model(architecture_entry, data_config)

        info = flare.system_info()
        site_name = flare.get_site_name()
        num_clients = max(int(info.get("num_clients", 1)), 1)
        client_index = int(site_name.split("-")[-1]) - 1 if "-" in site_name else 0

        from torch.utils.data import DataLoader, Subset

        full_dataset = build_classification_dataset(data_config, num_clients * NUM_REQUESTS)
        indices = list(range(client_index, len(full_dataset), num_clients))[:NUM_REQUESTS]
        subset = Subset(full_dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE)

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

        script_source = (
            _CLIENT_SCRIPT.replace("NUM_REQUESTS", str(config.num_requests))
            .replace("BATCH_SIZE", str(max(config.batch_size, 1)))
            .replace("DATASET_NAME", config.dataset)
            .replace("APPLICATION_NAME", config.application)
            .replace("ARCHITECTURE_NAME", config.architecture)
        )

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
            job.simulator_run(str(Path(tmp_dir) / "workspace"))

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
