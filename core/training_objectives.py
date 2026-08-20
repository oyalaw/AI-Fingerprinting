"""Pluggable per-architecture training objective for FL/distributed
training -- the real gap this project's own generalization work left
open: every fl_frameworks/*.py and distributed_frameworks/*.py adapter's
training loop used to hardcode `torch.nn.CrossEntropyLoss()` against a
`(input, scalar_class_label)` batch, which is exactly right for CNN/RNN/
Transformer's classification architectures but structurally wrong for
Autoencoder (no label at all -- the target IS the input), Diffusion
(no label either, and the "input" a training step needs is a real image
to noise, not the noise itself DDPM's own inference-time sampling starts
from), and AnomalyAutoencoder (same reconstruction target as Autoencoder,
but one attribute deeper -- its own forward() returns a reconstruction-
error score, not the reconstructed image, so the loss has to be computed
against self.autoencoder specifically). This module gives every adapter
one dispatch point instead of duplicating four near-identical
training-loop bodies across all six FL/distributed adapters.

A training step is `(model, batch, device) -> (loss, example_count)` --
every adapter's loop shrinks to:

    training_step = get_training_step(config.architecture)
    for batch in loader:
        loss, n = training_step(model, batch, device)
        loss.backward()

`batch` is always a tuple (1-tuple for reconstruction/denoising, 2-tuple
`(inputs, labels)` for classification) -- see core/training_data.py's
build_reconstruction_dataset()/build_denoising_dataset() for the dataset
side of this contract.
"""
import torch

from core.registry import ARCHITECTURES


def _classification_step(model, batch, device):
    """batch = (inputs, labels)."""
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    loss = torch.nn.functional.cross_entropy(output, labels)
    return loss, len(labels)


def _reconstruction_step(model, batch, device):
    """batch = (inputs,) -- Autoencoder reconstructs its own input, so
    the target IS the input, not a separately-loaded label."""
    (inputs,) = batch
    inputs = inputs.to(device)
    output = model(inputs)
    loss = torch.nn.functional.mse_loss(output, inputs)
    return loss, len(inputs)


def _anomaly_reconstruction_step(model, batch, device):
    """batch = (inputs,) -- architectures/autoencoder.py's
    _AnomalyAutoencoder wraps a plain autoencoder (self.autoencoder) but
    its own forward() returns the per-sample reconstruction-error score,
    not the reconstructed image (see that module's own docstring for
    why: applications/anomaly_detection.py's postprocess() needs that
    score directly, and Application.postprocess() never receives the
    original input to compute it there). Training this is the exact same
    MSE-against-its-own-input objective _reconstruction_step already
    expresses -- just computed one attribute deeper, against
    model.autoencoder directly, since model(inputs) itself returns a
    scalar score, not something comparable to inputs at all."""
    (inputs,) = batch
    inputs = inputs.to(device)
    reconstruction = model.autoencoder(inputs)
    loss = torch.nn.functional.mse_loss(reconstruction, inputs)
    return loss, len(inputs)


def _adversarial_step(model, batch, device):
    """batch = (real_images,) -- architectures/dcgan.py's DCGAN needs a
    real discriminator to train against at all, which doesn't exist on
    the plain served/registered generator (paradigm=inference only ever
    needs the generator) -- prepare_model_for_training() below attaches
    one as model.discriminator before this ever runs.

    A real, if simplified, adversarial training technique: a
    SIMULTANEOUS gradient update (both networks' losses summed into one
    scalar, one .backward() call) rather than the more common
    alternating G-then-D scheme -- a known GAN training variant
    (simultaneous SGD/Adam updates), not a fabricated proxy, chosen
    specifically so this still fits the exact same one-loss/one-backward/
    one-optimizer shape every other training_objective already uses:
    model.parameters() already covers both networks together (attaching
    an nn.Module as an attribute registers it as a real submodule), so
    no adapter needs its own special-cased two-optimizer alternation
    code. fake_images.detach() when scoring the discriminator on fakes
    is still real and necessary here -- without it the discriminator's
    loss term would backprop into the generator too, which the
    generator's own loss term (undetached) already does correctly on
    purpose."""
    (real_images,) = batch
    real_images = real_images.to(device)
    batch_size = real_images.shape[0]

    noise = torch.randn_like(real_images)
    fake_images = model(noise)

    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)

    d_real = model.discriminator(real_images)
    d_fake_detached = model.discriminator(fake_images.detach())
    d_fake_for_generator = model.discriminator(fake_images)

    loss_d_real = torch.nn.functional.binary_cross_entropy(d_real, real_labels)
    loss_d_fake = torch.nn.functional.binary_cross_entropy(d_fake_detached, fake_labels)
    loss_g = torch.nn.functional.binary_cross_entropy(d_fake_for_generator, real_labels)

    return loss_d_real + loss_d_fake + loss_g, batch_size


def _denoising_step(model, batch, device):
    """batch = (images,) real images (NOT the pure noise
    architectures/ddpm.py's own forward() samples from at inference --
    training needs a real image to add synthetic noise to and then ask
    the model to remove). Standard DDPM training procedure: sample a
    random timestep and a random noise tensor per example, compute x_t in
    closed form (no need to run the reverse loop), and have the model's
    own noise_predictor submodule predict the noise that was added --
    reusing architectures/ddpm.py's own _DDPMSampler.alphas_cumprod
    buffer and .noise_predictor submodule directly rather than
    duplicating the schedule math here.
    """
    (images,) = batch
    images = images.to(device)
    batch_size = images.shape[0]
    t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
    noise = torch.randn_like(images)
    alphas_cumprod_t = model.alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alphas_cumprod_t) * images + torch.sqrt(1.0 - alphas_cumprod_t) * noise
    predicted_noise = model.noise_predictor(x_t, t)
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    return loss, batch_size


_TRAINING_STEPS = {
    "classification": _classification_step,
    "reconstruction": _reconstruction_step,
    "anomaly_reconstruction": _anomaly_reconstruction_step,
    "adversarial": _adversarial_step,
    "denoising": _denoising_step,
}

# Which nn.Module attribute get_trainable_module()/set_trainable_module()
# below should reach into, for objectives whose *whole* registered model
# isn't the thing that actually gets a forward+backward pass during
# training (see each's own training step above for why). Objectives not
# listed here (classification, reconstruction) train through `model`
# itself -- unlisted is the common case, not an omission.
_SUBMODULE_ATTR_FOR_OBJECTIVE = {
    "denoising": "noise_predictor",
    "anomaly_reconstruction": "autoencoder",
}


def get_training_objective(architecture_name):
    """Every classification architecture (ResNet18/50, MobileNetV2, ViT,
    BERT, DistilBERT, LSTM, GRU, MLP) omits the training_objective
    metadata field entirely and implicitly defaults to "classification"
    here -- only Autoencoder/DDPM declare it explicitly (see each
    architecture's own registration)."""
    return ARCHITECTURES.get(architecture_name).meta.get("training_objective", "classification")


def get_training_step(architecture_name):
    objective = get_training_objective(architecture_name)
    return _TRAINING_STEPS[objective]


def is_classification(architecture_name):
    """core/classification_metrics.py's accuracy/precision/recall/F1 are
    only meaningful when the model outputs class logits -- FL adapters
    use this to decide whether to compute and log them at all."""
    return get_training_objective(architecture_name) == "classification"


def prepare_model_for_training(model, architecture_name):
    """Called once, right after building the model and before
    constructing any optimizer -- some objectives need the model object
    augmented with an extra submodule before training can start at all.
    Adversarial training needs a real discriminator attached
    (architectures/dcgan.py's attach_discriminator()), which doesn't
    exist on the plain served/registered generator paradigm=inference
    builds. Every other objective is a no-op here."""
    if get_training_objective(architecture_name) == "adversarial":
        from architectures.dcgan import attach_discriminator

        return attach_discriminator(model)
    return model


def get_trainable_module(model, architecture_name):
    """Which nn.Module a distributed_frameworks/*.py adapter should
    actually wrap in DistributedDataParallel/deepspeed.initialize()/
    FairScale's OSS -- not always `model` itself. Two real cases where
    it isn't (_SUBMODULE_ATTR_FOR_OBJECTIVE above): architectures/ddpm.py's
    top-level forward() implements a T-step *inference* sampling loop
    around one real trainable submodule (model.noise_predictor) --
    wrapping the whole sampler would try to differentiate through that
    non-training loop instead. architectures/autoencoder.py's
    AnomalyAutoencoder similarly wraps a plain autoencoder
    (model.autoencoder) but its own forward() returns a reconstruction-
    error *score*, not something a reconstruction loss can be computed
    against directly -- see _anomaly_reconstruction_step above. Every
    other training_objective's whole model IS what fit()s in one forward
    call, so this is just `model` for those."""
    attr = _SUBMODULE_ATTR_FOR_OBJECTIVE.get(get_training_objective(architecture_name))
    return getattr(model, attr) if attr else model


def set_trainable_module(model, architecture_name, wrapped_module):
    """Counterpart to get_trainable_module(): puts the now-wrapped
    module back where training_step()'s own attribute access expects to
    find it, for objectives in _SUBMODULE_ATTR_FOR_OBJECTIVE -- the rest
    of the outer object (model.alphas_cumprod/model.num_timesteps for
    denoising, nothing extra for anomaly_reconstruction) stays plain and
    unwrapped either way. For every other objective, the wrapped module
    simply replaces `model` outright, since there was no outer object
    distinct from what got wrapped."""
    attr = _SUBMODULE_ATTR_FOR_OBJECTIVE.get(get_training_objective(architecture_name))
    if attr:
        setattr(model, attr, wrapped_module)
        return model
    return wrapped_module
