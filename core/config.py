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

# Every fl_frameworks/*.py and distributed_frameworks/*.py adapter's
# training loop calls `model.state_dict()`/`.parameters()`/`.train()`
# directly on whatever `framework.load_model()` returns, which only a
# plain PyTorch nn.Module supports (confirmed:
# frameworks/openvino_adapter.py's load_model() returns an OpenVINOModel
# wrapper with none of those methods) -- so framework is locked to
# PyTorch. Architecture is locked to architectures that are genuinely
# classification-shaped (one scalar string label per independent sample,
# CrossEntropyLoss over class logits) -- confirmed directly, not just
# assumed, for all 8 of these: Image Classification (ResNet18/ResNet50/
# MobileNetV2/ViT, num_classes=10), Sentiment Analysis (BERT/DistilBERT,
# num_classes=2), Activity Recognition (LSTM/GRU, num_classes=6). Node
# Classification (GCN) does NOT fit -- confirmed structurally, not by
# policy: datasets/karate_club.py's samples() yields the same whole-graph
# input n times, each paired with a list of all 34 node labels at once
# (not a scalar), and GCN's forward pass is transductive (one graph in,
# all-node logits out per call) -- see core/training_data.py's docstring
# for the full finding. Every generative/structured-output application
# (Text Generation, Image Generation, Speech Recognition, Object
# Detection, Segmentation, Image Reconstruction) has no scalar
# classification label either and stays excluded the same way.
#
# Dataset compatibility isn't a second, separate flat list -- it's derived
# from the already-existing APPLICATIONS registry's own `datasets=[...]`
# metadata (applications/*.py's registration calls, the same list
# main.py's --interactive Application->Dataset filter already uses), so
# there's nothing here that could drift out of sync with it. A dataset
# still needs to actually fit the locked architecture's num_classes --
# core/training_data.py's build_classification_dataset() checks that
# directly against each architecture's own num_classes metadata (not a
# blanket constant), e.g. so ImageNet's 1000 classes correctly can't be
# paired with num_classes=10 architectures without a config.yaml error
# well before any training loop runs.
#
# core/training_data.py's build_classification_dataset() is what actually
# loads whichever dataset is selected via the DATASETS/APPLICATIONS
# registries (the same abstraction paradigm=inference uses), instead of
# every adapter hardcoding its own torchvision.datasets.CIFAR10 call the
# way they all used to.
#
# A single source of truth here, reused by both this module's validate()
# and main.py's --interactive locking, so the two can't drift apart.
FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES = (
    "ResNet18", "ResNet50", "MobileNetV2", "ViT",  # Image Classification
    "BERT", "DistilBERT",  # Sentiment Analysis
    "LSTM", "GRU", "MLP",  # Activity Recognition
)
FL_DISTRIBUTED_FRAMEWORK = "PyTorch"

# fl_frameworks/fedgraph_adapter.py is a deliberate, narrow exception to the
# constraint above: unlike every other FL/distributed adapter, it doesn't
# call build_classification_dataset()/.state_dict() on a model this project's
# own registries build -- it drives FedGraph's own real run_GC() pipeline
# end-to-end (FedGraph's own data_loader_GC_single() partitioning real MUTAG
# graphs across FedGraph's own Ray-actor trainers, FedGraph's own GIN model
# construction), confirmed directly to work this way, not routed through
# FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES's classification loader at all.
# architecture=GIN's own build() still returns a real, independently-usable
# PyTorch module for paradigm=inference (see architectures/gin.py) -- this
# tuple only controls what's additionally allowed for fl_framework=FedGraph.
FEDGRAPH_COMPATIBLE_ARCHITECTURES = ("GIN",)


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

    # Separate from capture -- see telemetry/resource_monitor.py's own
    # docstring for why host-side CPU/GPU/power/energy telemetry is kept
    # as its own artifact rather than folded into the attacker-facing
    # traffic features.
    resource_telemetry: bool = True
    resource_sample_interval_ms: int = 500

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
    client_index: int = 0  # FL: which partition of config.dataset this client trains on
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
            # fl_framework: FedGraph gets its own, narrower allowlist (just
            # GIN) instead of the general classification-architecture one --
            # confirmed directly this must be a *strict* per-fl_framework
            # choice, not a permissive "GIN or any of the 8" union: an
            # earlier version of this check allowed fl_framework: FedGraph
            # with e.g. architecture: ResNet18 to pass validation (ResNet18
            # is itself a valid FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES
            # entry) even though fl_frameworks/fedgraph_adapter.py's
            # run_server() ignores config.architecture/config.dataset
            # entirely and always runs its own real MUTAG/GIN pipeline --
            # silently mislabeling ground truth exactly the way this
            # project's own FL/distributed generalization work (see
            # README.md) already fixed once for hardcoded CIFAR10.
            architecture_allowlist = (
                FEDGRAPH_COMPATIBLE_ARCHITECTURES
                if self.fl_framework == "FedGraph"
                else FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES
            )
            if (self.architecture or "") not in architecture_allowlist:
                errors.append(
                    f"paradigm '{self.paradigm}'"
                    + (f" with fl_framework '{self.fl_framework}'" if self.fl_framework == "FedGraph" else "")
                    + f" only works with architecture(s) {architecture_allowlist} today -- every "
                    f"fl_frameworks/distributed_frameworks adapter's training loop expects a "
                    f"specific model shape its own data pipeline was built for, which "
                    f"'{self.architecture}' isn't."
                )
            elif APPLICATIONS.has(self.application):
                arch_application = ARCHITECTURES.get(self.architecture).meta.get("application")
                if arch_application and arch_application.lower() != (self.application or "").lower():
                    # Architecture<->application mismatches aren't checked
                    # anywhere else in validate() (a general, pre-existing
                    # gap for paradigm=inference too), but for FL/distributed
                    # specifically the consequences are worse than a wrong
                    # ground-truth label: e.g. architecture=BERT with
                    # application=Image Classification would crash deep
                    # inside _BertWrapper.forward() on a raw image tensor
                    # instead of the (3, seq_len) token stack it expects.
                    errors.append(
                        f"architecture '{self.architecture}' pairs with application "
                        f"'{arch_application}', not '{self.application}'."
                    )
                else:
                    # Dataset compatibility for FL/distributed reuses the
                    # same per-application datasets=[...] metadata everything
                    # else already relies on, rather than a second, separate
                    # list -- see FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES's
                    # comment above.
                    allowed_datasets = APPLICATIONS.get(self.application).meta.get("datasets") or ()
                    if allowed_datasets and (self.dataset or "") not in allowed_datasets:
                        errors.append(
                            f"application '{self.application}' only works with dataset(s) "
                            f"{tuple(allowed_datasets)} today, not '{self.dataset}' -- checked here "
                            f"specifically for paradigm '{self.paradigm}' (a mismatch would otherwise "
                            f"silently mislabel ground truth or crash inside the training loop; this "
                            f"isn't enforced for paradigm=inference, see README.md)."
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
