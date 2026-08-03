"""Ray Train -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Ray Train is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("Ray Train", implemented=False, organization='Anyscale')(build)
