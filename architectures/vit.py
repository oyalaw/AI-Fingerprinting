"""ViT -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("ViT is not yet implemented.")


ARCHITECTURES.register(
    "ViT", implemented=False, family="Transformer", framework="PyTorch"
)(build)
