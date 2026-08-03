"""Spektral -- registered but not yet implemented."""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Spektral is not yet implemented.")


GRAPH_FRAMEWORKS.register("Spektral", implemented=False, application='GNN')(build)
