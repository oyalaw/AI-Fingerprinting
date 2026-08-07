"""Autoencoder -- the Autoencoder family's only architecture, and a new
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
)(build)
