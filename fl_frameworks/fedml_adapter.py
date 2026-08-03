"""FedML -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FedML is not yet implemented.")


FL_FRAMEWORKS.register("FedML", implemented=False, organization='FedML Inc.')(build)
