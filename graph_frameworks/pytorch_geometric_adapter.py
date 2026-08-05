"""PyTorch Geometric -- real, verified implementation of a GCN classifier
equivalent to architectures/gcn.py's hand-rolled one, built with
`torch_geometric.nn.GCNConv` instead of raw `adjacency @ (features @ W)`.

**Honest disclosure on wiring**: this is NOT currently reachable by setting
`graph_framework: PyTorch Geometric` in a config. `architecture_entry.build()`
(the call every architecture module implements, see architectures/gcn.py)
only ever receives the selected `framework_adapter` -- never the full
`ExperimentConfig` -- so there's no way for GCN's build() to see which
graph_framework was selected and branch on it. Making that real would mean
threading `config` through `FrameworkAdapter.load_model()` and every
`architecture_entry.build()` call site across all 12 currently-implemented
framework adapters and all 13 architectures -- a real, separate refactor,
not something to fold in unprompted alongside one adapter. `core/config.py`
already validates `graph_framework` against this registry; this file makes
that validated value point at something real for the first time, rather
than adding a 4th framework that's equally disconnected from execution --
but "validated" and "actually used" remain two different things until that
refactor happens.

**The math, done correctly, not just running without erroring**: GCN's
input is `datasets/karate_club.py`'s (2, 34, 34) stack, channel 0 already
being the *fully symmetrically-normalized* adjacency (D^-1/2 (A+I) D^-1/2,
self-loops already baked in, via families/gnn's shared helper) -- not a
raw binary adjacency. `GCNConv` normalizes internally by default, which
would double-normalize an already-normalized input. Confirmed directly:
`torch_geometric.utils.dense_to_sparse` on a dense adjacency with a
nonzero diagonal correctly recovers both `edge_index` *and* the
self-loop-inclusive `edge_weight`; feeding that into
`GCNConv(..., normalize=False, add_self_loops=False)` applies the
propagation using exactly the precomputed weights, the same math the
hand-rolled version does, not a numerically different GCN variant that
happens to also produce (34, 2) logits.

torch_geometric is a pure-Python wheel (confirmed: `torch_geometric-*-py3-
none-any.whl`, no platform-specific build) -- lighter installation risk
than architectures/gcn.py's docstring speculated when it chose to hand-roll
GCN instead of depending on this package.
"""
import torch

from core.registry import GRAPH_FRAMEWORKS


class _PyGGCNClassifier(torch.nn.Module):
    def __init__(self, num_nodes=34, hidden_dim=16, num_classes=2):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(num_nodes, hidden_dim, normalize=False, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, num_classes, normalize=False, add_self_loops=False)
        self.relu = torch.nn.ReLU()

    def forward(self, stacked_input):
        from torch_geometric.utils import dense_to_sparse

        if stacked_input.dim() == 4:
            stacked_input = stacked_input.squeeze(0)
        adjacency = stacked_input[0]  # (34, 34), already normalized
        features = stacked_input[1]  # (34, 34) identity node features

        edge_index, edge_weight = dense_to_sparse(adjacency)
        h1 = self.relu(self.conv1(features, edge_index, edge_weight))
        return self.conv2(h1, edge_index, edge_weight)


class PyTorchGeometricAdapter:
    """Not a FrameworkAdapter (frameworks/base.py) -- graph_frameworks/ has
    no base.py of its own yet, since nothing in roles/client.py or
    roles/server.py consults this registry (see module docstring)."""

    def build_gcn(self, num_nodes=34, hidden_dim=16, num_classes=2):
        return _PyGGCNClassifier(num_nodes, hidden_dim, num_classes)


@GRAPH_FRAMEWORKS.register("PyTorch Geometric", implemented=True, organization="PyG Team")
def build_pytorch_geometric_adapter(**kwargs):
    return PyTorchGeometricAdapter()
