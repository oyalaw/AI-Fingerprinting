"""Shared classification training-data loader for paradigm=federated_learning
and paradigm=distributed_training. Every fl_frameworks/*.py and
distributed_frameworks/*.py adapter previously hardcoded its own
torchvision.datasets.CIFAR10 loader, ignoring config.dataset entirely --
this module replaces that with the same DATASETS/APPLICATIONS registry
abstraction paradigm=inference already uses (roles/client.py,
roles/standalone.py), so a config's dataset selection genuinely reflects
what traffic gets generated instead of silently mislabeling ground truth.

Deliberately scoped narrow: only valid for the architecture/application/
dataset combinations core/config.py's FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES
/FL_DISTRIBUTED_COMPATIBLE_DATASETS lock paradigm=federated_learning/
distributed_training to (Image Classification with num_classes=10
architectures, CIFAR10 or Synthetic datasets) -- see that module's
docstring for why this doesn't generalize further: Dataset.samples()
yields a human-readable label *string* for inference-time logging only,
there's no numeric-label or loss-function concept anywhere in the
Application interface, so genuinely different applications (text
generation, node classification, ...) would need that concept invented
from scratch, out of scope here.
"""
from core.registry import APPLICATIONS, DATASETS

_MAX_CLASSES = 10  # matches every FL/distributed-compatible architecture's num_classes=10


def build_classification_dataset(config, total_samples_needed):
    """Returns a torch.utils.data.TensorDataset of (input, label_index)
    pairs built from config.dataset via the DATASETS/APPLICATIONS
    registries. Callers own their own partitioning (Subset/stride-slice for
    the FL adapters, DistributedSampler for DDP/FairScale) -- this just
    replaces the data *source*, not each adapter's existing distributed
    mechanics.

    Numeric labels are derived from the *observed* set of Dataset.samples()
    string labels (sorted for a deterministic, if arbitrary, index
    assignment) rather than a per-dataset class table: every architecture
    this project drives is random-init, so there's no canonical
    class-index alignment to preserve, only a consistent one -- the same
    "traffic over accuracy" policy documented throughout architectures/*.py.
    """
    import torch
    from torch.utils.data import TensorDataset

    application = APPLICATIONS.get(config.application).build()
    dataset = DATASETS.get(config.dataset).build()

    samples = list(dataset.samples(total_samples_needed))
    if not samples:
        raise RuntimeError(
            f"dataset '{config.dataset}' produced no samples for "
            f"paradigm='{config.paradigm}' -- nothing to train on."
        )

    label_names = sorted({label for _, label in samples})
    if len(label_names) > _MAX_CLASSES:
        raise RuntimeError(
            f"dataset '{config.dataset}' produced {len(label_names)} distinct labels, "
            f"more than the {_MAX_CLASSES} classes every FL/distributed-compatible "
            f"architecture is built for (num_classes={_MAX_CLASSES}) -- see "
            f"core/config.py's FL_DISTRIBUTED_COMPATIBLE_DATASETS."
        )
    label_to_index = {label: index for index, label in enumerate(label_names)}

    inputs = torch.stack([application.preprocess(raw) for raw, _ in samples])
    targets = torch.tensor([label_to_index[label] for _, label in samples], dtype=torch.long)

    return TensorDataset(inputs, targets)
