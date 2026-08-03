"""PyTorch Mobile -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("PyTorch Mobile is not yet implemented.")


FRAMEWORKS.register("PyTorch Mobile", implemented=False, organization='Meta', platforms=['android', 'ios', 'ipados'])(build)
