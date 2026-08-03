"""PyTorch Geometric -- registered but not yet implemented."""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("PyTorch Geometric is not yet implemented.")


GRAPH_FRAMEWORKS.register("PyTorch Geometric", implemented=False, application='GNN')(build)
