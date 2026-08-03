"""MobileNetV2 -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("MobileNetV2 is not yet implemented.")


ARCHITECTURES.register(
    "MobileNetV2", implemented=False, family="CNN", framework="PyTorch"
)(build)
