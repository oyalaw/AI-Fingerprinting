"""StellarGraph -- registered but not yet implemented."""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("StellarGraph is not yet implemented.")


GRAPH_FRAMEWORKS.register("StellarGraph", implemented=False, application='GNN')(build)
