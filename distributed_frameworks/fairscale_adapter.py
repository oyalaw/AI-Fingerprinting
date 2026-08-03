"""FairScale -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FairScale is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("FairScale", implemented=False, organization='Meta')(build)
