"""ColossalAI -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ColossalAI is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("ColossalAI", implemented=False, organization='HPC AI Tech')(build)
