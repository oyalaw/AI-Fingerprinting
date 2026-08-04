"""GCN -- first and only GNN-family architecture, hand-rolled (Kipf &
Welling, 2017) two-layer propagation rather than a torch_geometric/DGL
dependency: `graph_frameworks/pytorch_geometric_adapter.py` etc. remain
stubs (their own pip wheels carry real platform-specific install risk,
the same category of problem TensorFlow's Python-version wheel gap hit
elsewhere in this project) -- a GCN layer is simple enough
(`adjacency @ (features @ W)`) to implement directly in plain PyTorch,
avoiding that risk entirely for this one architecture.

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


def build(framework_adapter):
    return _GCNClassifier()


ARCHITECTURES.register(
    "GCN", implemented=True, family="GNN", framework="PyTorch", input_shape=(2, 34, 34)
)(build)
