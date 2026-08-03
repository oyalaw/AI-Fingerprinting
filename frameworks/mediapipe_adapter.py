"""MediaPipe -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("MediaPipe is not yet implemented.")


FRAMEWORKS.register("MediaPipe", implemented=False, organization='Google', platforms=['android', 'ios', 'ipados', 'macos'])(build)
