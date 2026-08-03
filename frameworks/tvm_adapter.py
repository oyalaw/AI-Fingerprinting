"""TVM -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TVM is not yet implemented.")


FRAMEWORKS.register("TVM", implemented=False, organization='Apache', platforms=['windows', 'linux', 'macos', 'android', 'jetson'])(build)
