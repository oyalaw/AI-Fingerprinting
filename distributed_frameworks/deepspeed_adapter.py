"""DeepSpeed -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DeepSpeed is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("DeepSpeed", implemented=False, organization='Microsoft')(build)
