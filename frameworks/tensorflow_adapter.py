"""TensorFlow -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorFlow is not yet implemented.")


FRAMEWORKS.register("TensorFlow", implemented=False, organization='Google', platforms=['windows', 'linux', 'macos', 'jetson'])(build)
