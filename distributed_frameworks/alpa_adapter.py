"""Alpa -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Alpa is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("Alpa", implemented=False, organization='Stanford')(build)
