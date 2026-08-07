"""GCN -- first GNN-family architecture. Default build is a hand-rolled
(Kipf & Welling, 2017) two-layer propagation in plain PyTorch
(`adjacency @ (features @ W)`); when `graph_framework: PyTorch Geometric`
is set in the config, build() now dispatches to
graph_frameworks/pytorch_geometric_adapter.py's real GCNConv-based
classifier instead -- see that module's docstring for how its math was
verified equivalent to the hand-rolled version, not just "also produces
the right shape."

Random init, small (hidden_dim=16) -- consistent with every other
architecture here. Input: the (2, 34, 34) stacked tensor
datasets/karate_club.py produces (channel 0 = normalized adjacency,
channel 1 = identity node features); output: (34, 2) per-node logits
over the graph's 2 real factions ("Mr. Hi" vs "Officer").
"""
import torch

from core.registry import ARCHITECTURES


class _GCNClassifier(torch.nn.Module):
    def __init__(self, num_nodes=34, hidden_dim=16, num_classes=2):
        super().__init__()
        self.layer1 = torch.nn.Linear(num_nodes, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, stacked_input):
        if stacked_input.dim() == 4:
            stacked_input = stacked_input.squeeze(0)
        adjacency = stacked_input[0]  # (34, 34)
        features = stacked_input[1]  # (34, 34)

        h1 = self.relu(adjacency @ self.layer1(features))
        h2 = adjacency @ self.layer2(h1)
        return h2


def build(framework_adapter, config):
    graph_framework = getattr(config, "graph_framework", None)
    if graph_framework:
        from core.registry import GRAPH_FRAMEWORKS

        # entry.build() itself raises the standard NotImplementedError for
        # a stub graph_framework -- no special-casing needed here for that.
        adapter = GRAPH_FRAMEWORKS.get(graph_framework).build()
        if not hasattr(adapter, "build_gcn"):
            raise RuntimeError(
                f"graph_framework '{graph_framework}' has no GCN implementation "
                f"to dispatch to (see architectures/gcn.py)."
            )
        return adapter.build_gcn()
    return _GCNClassifier()


ARCHITECTURES.register(
    "GCN",
    implemented=True,
    family="GNN",
    framework="PyTorch",
    application="Node Classification",
    input_shape=(2, 34, 34),
)(build)
