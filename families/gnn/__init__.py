"""GNN family: shared adjacency-normalization helper reused by every GNN
architecture (GCN, ...). This is the Kipf & Welling (2017) propagation
rule's normalization step: D^-1/2 (A + I) D^-1/2 -- self-loops added so
each node's own features survive propagation, then symmetric degree
normalization so high-degree nodes don't dominate.
"""
import numpy as np

from core.registry import FAMILIES


def normalize_adjacency(adjacency):
    """(N, N) binary adjacency -> (N, N) symmetrically-normalized adjacency
    with self-loops, as a float32 numpy array."""
    num_nodes = adjacency.shape[0]
    adjacency_with_self_loops = adjacency + np.eye(num_nodes, dtype=adjacency.dtype)
    degree = adjacency_with_self_loops.sum(axis=1)
    # `out=` is required alongside `where=` -- without it, positions where
    # the condition is False are left as uninitialized memory, not zero
    # (confirmed directly: numpy warns "expect uninitialized memory in
    # output"). Harmless today since Karate Club has no degree-0 node, but
    # that's luck, not correctness, for any future graph that does.
    degree_inv_sqrt = np.zeros_like(degree)
    np.power(degree, -0.5, out=degree_inv_sqrt, where=degree > 0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    return (D_inv_sqrt @ adjacency_with_self_loops @ D_inv_sqrt).astype(np.float32)


FAMILIES.add("GNN", implemented=True, description="Graph Neural Networks")
