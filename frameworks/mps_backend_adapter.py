"""MPS Backend -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("MPS Backend is not yet implemented.")


FRAMEWORKS.register("MPS Backend", implemented=False, organization='Apple', platforms=['macos'])(build)
