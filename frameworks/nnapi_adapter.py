"""NNAPI -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("NNAPI is not yet implemented.")


FRAMEWORKS.register("NNAPI", implemented=False, organization='Google', platforms=['android'])(build)
