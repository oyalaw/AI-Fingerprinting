"""FedLab -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FedLab is not yet implemented.")


FL_FRAMEWORKS.register("FedLab", implemented=False, organization='SMILELab')(build)
