"""DCGAN generator (Radford, Metz & Chintala, 2015) -- first and only
GAN-family architecture. Only the generator is registered, not a paired
discriminator: this project's registry names the model actually SERVED
over the wire during a paradigm=inference request/response loop (the
same reasoning architectures/ddpm.py gives for registering only its
sampler, not a training loop) -- a discriminator is a training-time-only
component with nothing to serve at inference.

Reuses applications/image_generation.py unchanged: preprocess() already
produces (3, 32, 32) Gaussian noise for architectures/ddpm.py, and a
GAN's generator input genuinely is noise too, just conventionally a
smaller latent vector rather than a full noise image. Rather than change
the shared application to emit two different noise shapes depending on
which architecture asked, this generator does that reshape internally
(flatten the (3, 32, 32) noise this project's pipeline already provides,
project down to a real 128-dim latent, then run the standard DCGAN
ConvTranspose2d upsampling stack from there) -- keeps input_shape=(3, 32,
32) consistent with every other image-in-image-out architecture here
(DDPM, Autoencoder), and needs zero changes to shared application code.

Real DCGAN generator shape, verified by hand: ConvTranspose2d(k=4,s=1,p=0)
1x1->4x4, then three ConvTranspose2d(k=4,s=2,p=1) stages 4x4->8x8->16x16->
32x32 -- the paper's standard 32x32 recipe (same output resolution as
every CIFAR10-scale architecture in this project). Random init, no
training -- consistent with every other architecture here (BERT: 2
layers, GCN: hidden_dim=16): realistic traffic is the point, not sample
quality, so an untrained generator's output being visual noise is
expected, not a bug.
"""
import torch

from core.registry import ARCHITECTURES

_LATENT_DIM = 128


class _Generator(torch.nn.Module):
    def __init__(self, noise_shape=(3, 32, 32), latent_dim=_LATENT_DIM):
        super().__init__()
        noise_size = noise_shape[0] * noise_shape[1] * noise_shape[2]
        self.project = torch.nn.Linear(noise_size, latent_dim)
        self.upsample = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),  # 1x1 -> 4x4
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),  # 4x4 -> 8x8
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),  # 8x8 -> 16x16
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid(),  # 16x16 -> 32x32, pixel values in [0, 1]
        )

    def forward(self, noise_image):
        if noise_image.dim() == 3:
            noise_image = noise_image.unsqueeze(0)
        batch_size = noise_image.shape[0]
        latent = self.project(noise_image.reshape(batch_size, -1))
        latent = latent.view(batch_size, -1, 1, 1)
        return self.upsample(latent)


def build(framework_adapter, config):
    return _Generator()


ARCHITECTURES.register(
    "DCGAN",
    implemented=True,
    family="GAN",
    framework="PyTorch",
    application="Image Generation",
    input_shape=(3, 32, 32),
)(build)
