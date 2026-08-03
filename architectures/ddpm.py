"""DDPM -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("DDPM is not yet implemented.")


ARCHITECTURES.register(
    "DDPM", implemented=False, family="Diffusion", framework="PyTorch"
)(build)
