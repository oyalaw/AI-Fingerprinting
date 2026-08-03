"""Metal Performance Shaders -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Metal Performance Shaders is not yet implemented.")


FRAMEWORKS.register("Metal Performance Shaders", implemented=False, organization='Apple', platforms=['ios', 'ipados', 'macos'])(build)
