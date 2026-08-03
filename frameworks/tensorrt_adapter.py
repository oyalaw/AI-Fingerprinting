"""TensorRT -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorRT is not yet implemented.")


FRAMEWORKS.register("TensorRT", implemented=False, organization='NVIDIA', platforms=['jetson'])(build)
