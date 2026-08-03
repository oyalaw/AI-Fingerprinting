"""YOLOv8 -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("YOLOv8 is not yet implemented.")


ARCHITECTURES.register(
    "YOLOv8", implemented=False, family="CNN", framework="PyTorch"
)(build)
