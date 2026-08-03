"""SNPE -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("SNPE is not yet implemented.")


FRAMEWORKS.register("SNPE", implemented=False, organization='Qualcomm', platforms=['android'])(build)
