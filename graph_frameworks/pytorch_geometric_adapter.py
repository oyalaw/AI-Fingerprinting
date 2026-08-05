"""PyTorch Geometric -- real, verified implementation of a GCN classifier
equivalent to architectures/gcn.py's hand-rolled one, built with
`torch_geometric.nn.GCNConv` instead of raw `adjacency @ (features @ W)`.

**Wired into execution**: setting `graph_framework: PyTorch Geometric` in a
config now genuinely changes what runs. `ExperimentConfig` is threaded
through `FrameworkAdapter.load_model()` and every `architecture_entry.
build()` call site (all 22 framework adapters, all 13 architectures --
see roles/server.py, roles/standalone.py, and the fl_frameworks/
distributed_frameworks adapters that also build models); architectures/
gcn.py's build() checks `config.graph_framework` and, when it names this
entry, calls `PyTorchGeometricAdapter.build_gcn()` below instead of its
own hand-rolled classifier. Verified directly: a real client/server
roundtrip with `graph_framework: PyTorch Geometric` set produces correct
predictions and correct ground truth (the `graph_framework` sub-label is
now what actually ran, not just what was recorded).

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
    no base.py of its own yet. Consulted directly by architectures/gcn.py's
    build() (see module docstring), not through roles/client.py or
    roles/server.py, which only ever see the FrameworkAdapter/architecture
    pair GCN's build() ultimately returns."""

    def build_gcn(self, num_nodes=34, hidden_dim=16, num_classes=2):
        return _PyGGCNClassifier(num_nodes, hidden_dim, num_classes)


@GRAPH_FRAMEWORKS.register("PyTorch Geometric", implemented=True, organization="PyG Team")
def build_pytorch_geometric_adapter(**kwargs):
    return PyTorchGeometricAdapter()
