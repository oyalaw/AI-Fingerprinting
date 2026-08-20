"""Autoencoder -- the Autoencoder family's architectures, and a new
addition rather than an existing stub: the original 11-architecture list
this project's taxonomy specifies never actually paired anything with the
Autoencoder family (6 families were listed, but only 5 had a matching
architecture) -- the same kind of well-justified gap-fill as
datasets/karate_club.py was for the GNN family.

Small conv encoder/decoder (16->32 channels, stride-2 downsampling then
transposed-conv upsampling) -- confirmed directly the output shape exactly
matches the (3, 32, 32) input and the Sigmoid decoder output stays
correctly bounded in [0, 1] before writing this file. Random init, no
training loop (consistent with every other architecture here): the point
is realistic reconstruction-shaped traffic, not reconstruction quality.

AnomalyAutoencoder (below) is the family's second architecture, reusing
_AutoencoderModel by composition rather than duplicating it: the standard
reconstruction-based anomaly-detection technique, where the model's own
per-sample reconstruction error IS the output, not the reconstructed
image. This exists specifically to fill the gap
applications/image_reconstruction.py's own docstring already flags:
Application.postprocess() only ever sees the model's output, never the
original input, so it structurally cannot compute "how well did this
reconstruct" on its own. Computing the error inside the architecture's
own forward() instead -- the same precedent architectures/ddpm.py already
sets (a registered architecture's forward() doing real task-specific
computation beyond one plain layer stack) -- sidesteps that gap entirely
without changing the shared Application interface every other
architecture/application pair here also relies on.
"""
import torch

from core.registry import ARCHITECTURES


class _AutoencoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32 -> 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16 -> 8
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build(framework_adapter, config):
    return _AutoencoderModel()


ARCHITECTURES.register(
    "Autoencoder",
    implemented=True,
    family="Autoencoder",
    framework="PyTorch",
    application="Image Reconstruction",
    input_shape=(3, 32, 32),
    # core/training_objectives.py dispatches on this for FL/distributed
    # training -- reconstruction loss (MSE against the model's own input),
    # not classification's CrossEntropyLoss-against-a-label. Every
    # classification architecture omits this field and implicitly
    # defaults to "classification" (see that module's own docstring).
    training_objective="reconstruction",
)(build)


class _AnomalyAutoencoder(torch.nn.Module):
    """Wraps _AutoencoderModel (composition, not duplication -- same
    encoder/decoder weights shape, freshly random-init here) and returns
    the per-sample reconstruction-error score itself as the output,
    instead of the reconstructed image. applications/anomaly_detection.py's
    postprocess() thresholds this score directly into a normal/anomalous
    decision -- see this module's own top-of-file docstring for why the
    error has to be computed here rather than in postprocess()."""

    def __init__(self):
        super().__init__()
        self.autoencoder = _AutoencoderModel()

    def forward(self, x):
        reconstruction = self.autoencoder(x)
        # Per-sample MSE reconstruction error -- mean over channel/height/
        # width, keeping the batch dimension, so the output is one scalar
        # score per input image rather than a full per-pixel error map.
        return torch.mean((reconstruction - x) ** 2, dim=(1, 2, 3))


def build_anomaly_detector(framework_adapter, config):
    return _AnomalyAutoencoder()


ARCHITECTURES.register(
    "AnomalyAutoencoder",
    implemented=True,
    family="Autoencoder",
    framework="PyTorch",
    application="Anomaly Detection",
    input_shape=(3, 32, 32),
    # Deliberately NOT wired into FL/distributed training (no
    # training_objective set, so core/training_objectives.py would treat
    # it as "classification" if it were ever added there by mistake) --
    # reconstruction-error-based anomaly detection is a threshold decision
    # on top of reconstruction, not its own distinct trainable objective;
    # it would train via the exact same "reconstruction" objective the
    # plain Autoencoder already has, on the *inner* self.autoencoder
    # submodule specifically, which core/training_objectives.py doesn't
    # yet know to reach into. Add that mapping explicitly if FL support
    # for this one is wanted later, rather than let it silently fall
    # through to the wrong default.
)(build_anomaly_detector)
