"""FedJAX -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FedJAX is not yet implemented.")


FL_FRAMEWORKS.register("FedJAX", implemented=False, organization='Google')(build)
