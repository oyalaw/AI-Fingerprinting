"""CoreML -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("CoreML is not yet implemented.")


FRAMEWORKS.register("CoreML", implemented=False, organization='Apple', platforms=['ios', 'ipados', 'macos'])(build)
