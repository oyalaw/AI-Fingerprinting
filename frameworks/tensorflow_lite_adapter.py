"""TensorFlow Lite -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorFlow Lite is not yet implemented.")


FRAMEWORKS.register("TensorFlow Lite", implemented=False, organization='Google', platforms=['android', 'ios', 'ipados', 'linux', 'windows', 'macos', 'jetson'])(build)
