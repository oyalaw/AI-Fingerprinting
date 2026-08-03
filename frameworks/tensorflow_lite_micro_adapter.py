"""TensorFlow Lite Micro -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorFlow Lite Micro is not yet implemented.")


FRAMEWORKS.register("TensorFlow Lite Micro", implemented=False, organization='Google', platforms=['android'])(build)
