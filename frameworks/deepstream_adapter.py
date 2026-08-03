"""DeepStream -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DeepStream is not yet implemented.")


FRAMEWORKS.register("DeepStream", implemented=False, organization='NVIDIA', platforms=['jetson'])(build)
