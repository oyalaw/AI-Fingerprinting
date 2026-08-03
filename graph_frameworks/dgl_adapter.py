"""DGL -- registered but not yet implemented."""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DGL is not yet implemented.")


GRAPH_FRAMEWORKS.register("DGL", implemented=False, application='GNN')(build)
