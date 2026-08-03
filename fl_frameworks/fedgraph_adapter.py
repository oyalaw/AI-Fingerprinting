"""FedGraph -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FedGraph is not yet implemented.")


FL_FRAMEWORKS.register("FedGraph", implemented=False, organization='Rutgers University')(build)
