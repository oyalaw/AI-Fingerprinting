"""ResNet50 -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("ResNet50 is not yet implemented.")


ARCHITECTURES.register(
    "ResNet50", implemented=False, family="CNN", framework="PyTorch"
)(build)
