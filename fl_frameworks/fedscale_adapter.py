"""FedScale -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FedScale is not yet implemented.")


FL_FRAMEWORKS.register("FedScale", implemented=False, organization='Michigan State University')(build)
