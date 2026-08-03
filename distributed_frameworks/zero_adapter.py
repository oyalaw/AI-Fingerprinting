"""ZeRO -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ZeRO is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("ZeRO", implemented=False, organization='Microsoft')(build)
