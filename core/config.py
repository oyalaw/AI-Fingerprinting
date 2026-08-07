import dataclasses
import pathlib
import typing

import yaml

from core.registry import (
    APPLICATIONS,
    ARCHITECTURES,
    CV_FRAMEWORKS,
    DATASETS,
    DEVICES,
    DIFFUSION_FRAMEWORKS,
    DISTRIBUTED_FRAMEWORKS,
    FAMILIES,
    FL_FRAMEWORKS,
    FRAMEWORKS,
    GRAPH_FRAMEWORKS,
    LLM_FRAMEWORKS,
    SPEECH_FRAMEWORKS,
    TRANSPORTS,
    discover_all,
)

VALID_PARADIGMS = ("inference", "federated_learning", "distributed_training")
VALID_ROLES = ("client", "server", "standalone")

# Every fl_frameworks/*.py and distributed_frameworks/*.py adapter
# hardcodes its own data loader to torchvision.datasets.CIFAR10 -- none of
# them read config.dataset at all (confirmed directly: grepped every
# adapter's data-loading code) -- and each one's training loop calls
# `model.state_dict()`/`.parameters()`/`.train()` directly on whatever
# `framework.load_model()` returns, which only a plain PyTorch nn.Module
# supports (confirmed: frameworks/openvino_adapter.py's load_model()
# returns an OpenVINOModel wrapper with none of those methods). A single
# source of truth here, reused by both this module's validate() and
# main.py's --interactive locking, so the two can't drift apart.
FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES = ("ResNet18", "ResNet50", "MobileNetV2", "ViT")
FL_DISTRIBUTED_DATASET = "CIFAR10"
FL_DISTRIBUTED_FRAMEWORK = "PyTorch"


@dataclasses.dataclass
class ExperimentConfig:
    paradigm: str
    role: str
    device: str
    framework: str
    family: str
    architecture: str
    application: str
    dataset: str
    transport: str = "tcp"

    host: str = "127.0.0.1"
    port: int = 8765
    num_requests: int = 50
    batch_size: int = 1

    capture: bool = True
    capture_interface: typing.Optional[str] = None

    tls_cert: typing.Optional[str] = None
    tls_key: typing.Optional[str] = None

    results_dir: str = "experiments/results"

    fl_framework: typing.Optional[str] = None
    distributed_framework: typing.Optional[str] = None
    llm_framework: typing.Optional[str] = None
    cv_framework: typing.Optional[str] = None
    speech_framework: typing.Optional[str] = None
    graph_framework: typing.Optional[str] = None
    diffusion_framework: typing.Optional[str] = None

    num_clients: int = 2
    num_rounds: int = 3
    client_index: int = 0  # FL: which CIFAR10 partition this client trains on
    worker_rank: int = 1  # distributed training: this process's rank (coordinator is always rank 0)

    def validate(self):
        errors = []

        def check(registry, value, field_name, required=True):
            if value is None:
                if required:
                    errors.append(f"{field_name} is required for this paradigm/application")
                return
            if not registry.has(value):
                errors.append(
                    f"Unknown {field_name} '{value}'. Run `python main.py --list` to see valid options."
                )
                return
            entry = registry.get(value)
            if not entry.implemented:
                errors.append(
                    f"{field_name} '{value}' is registered but not yet implemented "
                    f"(see {entry.factory.__module__})."
                )

        if self.paradigm not in VALID_PARADIGMS:
            errors.append(f"paradigm must be one of {VALID_PARADIGMS}, got '{self.paradigm}'")
        if self.role not in VALID_ROLES:
            errors.append(f"role must be one of {VALID_ROLES}, got '{self.role}'")

        check(DEVICES, self.device, "device")
        check(FRAMEWORKS, self.framework, "framework")
        check(FAMILIES, self.family, "family")
        check(ARCHITECTURES, self.architecture, "architecture")
        check(APPLICATIONS, self.application, "application")
        check(DATASETS, self.dataset, "dataset")
        check(TRANSPORTS, self.transport, "transport")

        if self.paradigm == "federated_learning":
            check(FL_FRAMEWORKS, self.fl_framework, "fl_framework")
        if self.paradigm == "distributed_training":
            check(DISTRIBUTED_FRAMEWORKS, self.distributed_framework, "distributed_framework")

        if self.paradigm in ("federated_learning", "distributed_training"):
            # Real implementation constraint, not a registry-compatibility
            # one -- see FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES's comment
            # above for why. Framework/dataset mismatches here would
            # otherwise either crash deep inside the adapter (non-PyTorch
            # framework) or silently mislabel ground truth (a dataset other
            # than CIFAR10 that's never actually used) -- catching both
            # here, at the same place every other compatibility error is
            # caught, rather than leaving it as a runtime surprise.
            if (self.framework or "").lower() != FL_DISTRIBUTED_FRAMEWORK.lower():
                errors.append(
                    f"paradigm '{self.paradigm}' only works with framework "
                    f"'{FL_DISTRIBUTED_FRAMEWORK}' today -- every fl_frameworks/distributed_frameworks "
                    f"adapter calls .state_dict()/.parameters()/.train() directly on the loaded model, "
                    f"which only a plain PyTorch nn.Module supports, not '{self.framework}'."
                )
            if (self.architecture or "") not in FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES:
                errors.append(
                    f"paradigm '{self.paradigm}' only works with architecture(s) "
                    f"{FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES} today -- every fl_frameworks/"
                    f"distributed_frameworks adapter's training loop expects CIFAR10-shaped "
                    f"10-class classification output, which '{self.architecture}' doesn't produce."
                )
            if (self.dataset or "").lower() != FL_DISTRIBUTED_DATASET.lower():
                errors.append(
                    f"paradigm '{self.paradigm}' only works with dataset "
                    f"'{FL_DISTRIBUTED_DATASET}' today -- every fl_frameworks/distributed_frameworks "
                    f"adapter hardcodes its own CIFAR10 data loader and never reads config.dataset, "
                    f"so setting dataset: '{self.dataset}' would silently mislabel ground truth "
                    f"rather than actually being used."
                )

        application_lower = (self.application or "").lower()
        family_lower = (self.family or "").lower()

        if application_lower == "text generation" and self.llm_framework:
            check(LLM_FRAMEWORKS, self.llm_framework, "llm_framework", required=False)
        if application_lower in ("object detection", "segmentation") and self.cv_framework:
            check(CV_FRAMEWORKS, self.cv_framework, "cv_framework", required=False)
        if application_lower == "speech recognition" and self.speech_framework:
            check(SPEECH_FRAMEWORKS, self.speech_framework, "speech_framework", required=False)
        if family_lower == "gnn" and self.graph_framework:
            check(GRAPH_FRAMEWORKS, self.graph_framework, "graph_framework", required=False)
        if family_lower == "diffusion" and self.diffusion_framework:
            check(DIFFUSION_FRAMEWORKS, self.diffusion_framework, "diffusion_framework", required=False)

        if not errors and ARCHITECTURES.has(self.architecture):
            arch_entry = ARCHITECTURES.get(self.architecture)
            arch_family = arch_entry.meta.get("family")
            arch_framework = arch_entry.meta.get("framework")
            also_supports = arch_entry.meta.get("also_supports") or []
            compatible_frameworks = {arch_framework} | set(also_supports) if arch_framework else set(also_supports)

            if arch_family and arch_family.lower() != (self.family or "").lower():
                errors.append(
                    f"architecture '{self.architecture}' belongs to family "
                    f"'{arch_family}', not '{self.family}'"
                )
            if compatible_frameworks and (self.framework or "").lower() not in {
                f.lower() for f in compatible_frameworks
            }:
                errors.append(
                    f"architecture '{self.architecture}' supports framework(s) "
                    f"{sorted(compatible_frameworks)}, not '{self.framework}'"
                )

        if errors:
            raise ValueError("Invalid experiment config:\n  - " + "\n  - ".join(errors))


def load_config(path):
    discover_all()
    data = yaml.safe_load(pathlib.Path(path).read_text())
    config = ExperimentConfig(**data)
    config.validate()
    return config
