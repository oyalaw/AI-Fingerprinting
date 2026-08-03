"""Horovod -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Horovod is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("Horovod", implemented=False, organization='Uber')(build)
