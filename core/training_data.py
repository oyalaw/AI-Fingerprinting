"""Shared classification training-data loader for paradigm=federated_learning
and paradigm=distributed_training. Every fl_frameworks/*.py and
distributed_frameworks/*.py adapter previously hardcoded its own
torchvision.datasets.CIFAR10 loader, ignoring config.dataset entirely --
this module replaces that with the same DATASETS/APPLICATIONS registry
abstraction paradigm=inference already uses (roles/client.py,
roles/standalone.py), so a config's dataset selection genuinely reflects
what traffic gets generated instead of silently mislabeling ground truth.

Scoped to applications that are genuinely classification-shaped: one
scalar string label per independent sample, model output is class
logits, trained with CrossEntropyLoss. Image Classification (ResNet18/
ResNet50/MobileNetV2/ViT), Sentiment Analysis (BERT/DistilBERT), and
Activity Recognition (LSTM/GRU) all fit this exactly -- confirmed
directly for each: Application.preprocess() always returns a fixed-shape
tensor per sample, and every compatible dataset's samples() yields
exactly as many distinct label strings as its paired architecture's
num_classes (confirmed via architectures/*.py's own num_classes=
metadata, e.g. 10 for CIFAR10/Synthetic, 2 for IMDB/SST2's positive/
negative, 6 for UCI HAR's activity labels).

Node Classification (GCN) does NOT fit and was confirmed structurally
incompatible, not just excluded by policy: datasets/karate_club.py's
samples(n) yields the *same* whole-graph array n times, each paired with
the full list of all 34 node labels at once (a list, not a scalar --
would crash the label_names set-comprehension below outright), and GCN's
forward pass is transductive (one graph in, all-node logits out per
call, no batch-of-independent-samples concept at all). Every other
generative/structured-output application (Text Generation, Speech
Recognition, Object Detection, Segmentation) still doesn't fit either.

Autoencoder (Image Reconstruction) and DDPM (Image Generation) DO now
have their own loaders below -- build_reconstruction_dataset() and
build_denoising_dataset() -- once core/training_objectives.py gave every
FL/distributed adapter somewhere to dispatch a non-classification loss
to. build_training_dataset() is the one entry point adapters should
actually call; it picks whichever of the three loaders below matches
config.architecture's own training_objective metadata, so adapter code
doesn't need to know which shape a given architecture needs. See
core/config.py's FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES for the enforced
set.
"""
from core.registry import APPLICATIONS, ARCHITECTURES, DATASETS
from core.training_objectives import get_training_objective


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

    The observed label count is checked against config.architecture's own
    num_classes metadata (not a blanket constant) -- catches e.g.
    accidentally pairing a 2-class dataset with a 6-class architecture
    precisely, rather than a one-size-fits-all "≤10" check that would
    silently let such a mismatch through.
    """
    import torch
    from torch.utils.data import TensorDataset

    application = APPLICATIONS.get(config.application).build()
    dataset = DATASETS.get(config.dataset).build()
    num_classes = ARCHITECTURES.get(config.architecture).meta.get("num_classes")

    samples = list(dataset.samples(total_samples_needed))
    if not samples:
        raise RuntimeError(
            f"dataset '{config.dataset}' produced no samples for "
            f"paradigm='{config.paradigm}' -- nothing to train on."
        )

    label_names = sorted({label for _, label in samples})
    if num_classes and len(label_names) > num_classes:
        raise RuntimeError(
            f"dataset '{config.dataset}' produced {len(label_names)} distinct labels, "
            f"more than architecture '{config.architecture}' is built for "
            f"(num_classes={num_classes}) -- see core/config.py's "
            f"FL_DISTRIBUTED_COMPATIBLE_ARCHITECTURES."
        )
    label_to_index = {label: index for index, label in enumerate(label_names)}

    inputs = torch.stack([application.preprocess(raw) for raw, _ in samples])
    targets = torch.tensor([label_to_index[label] for _, label in samples], dtype=torch.long)

    return TensorDataset(inputs, targets)


def build_reconstruction_dataset(config, total_samples_needed):
    """Autoencoder: returns a TensorDataset of (input,) 1-tuples -- the
    reconstruction target IS the input (core/training_objectives.py's
    _reconstruction_step computes MSE against it directly), so there's no
    second tensor to build the way build_classification_dataset() builds
    labels. Goes through Application.preprocess() normally (unlike
    build_denoising_dataset() below): Image Reconstruction's preprocess()
    already returns the real image tensor DDPM's Image Generation
    deliberately doesn't (see that function's own docstring for why)."""
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

    inputs = torch.stack([application.preprocess(raw) for raw, _ in samples])
    return TensorDataset(inputs)


def build_denoising_dataset(config, total_samples_needed):
    """DDPM: returns a TensorDataset of (image,) 1-tuples -- real images
    to add noise to during training, deliberately NOT going through
    Application.preprocess() the way every other loader here does.
    applications/image_generation.py's preprocess() is inference-shaped on
    purpose (a generation *request* genuinely has no real input, only
    noise), which is exactly wrong for training: DDPM's denoising
    objective needs the real image so it has something to noise and then
    learn to recover. Loads config.dataset directly and applies the same
    HWC-uint8 -> CHW-float01 conversion families/autoencoder's
    scale_to_unit_range() already provides for Image Reconstruction,
    rather than duplicating that conversion here."""
    import torch
    from torch.utils.data import TensorDataset

    from families.autoencoder import scale_to_unit_range

    dataset = DATASETS.get(config.dataset).build()
    samples = list(dataset.samples(total_samples_needed))
    if not samples:
        raise RuntimeError(
            f"dataset '{config.dataset}' produced no samples for "
            f"paradigm='{config.paradigm}' -- nothing to train on."
        )

    inputs = torch.stack([scale_to_unit_range(raw) for raw, _ in samples])
    return TensorDataset(inputs)


_DATASET_BUILDERS = {
    "classification": build_classification_dataset,
    "reconstruction": build_reconstruction_dataset,
    "denoising": build_denoising_dataset,
}


def build_training_dataset(config, total_samples_needed):
    """The one entry point fl_frameworks/*.py and distributed_frameworks/*.py
    adapters should call -- dispatches to whichever loader above matches
    config.architecture's own training_objective metadata (core/
    training_objectives.py), so adapter code doesn't need its own
    if/elif over architecture names."""
    objective = get_training_objective(config.architecture)
    return _DATASET_BUILDERS[objective](config, total_samples_needed)
