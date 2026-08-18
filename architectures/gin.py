"""GIN (Graph Isomorphism Network, Xu et al. 2019) -- second GNN-family
architecture, paired with the new Graph Classification application
(architectures/gcn.py/Node Classification is transductive -- one graph,
classify every node -- structurally different from graph classification,
where each *graph* is one independent sample; see
datasets/mutag.py's docstring for why that needed a real second
dataset/application, not just a second architecture).

Built with `torch_geometric.nn.GINConv` directly (real, verified layers --
same precedent as graph_frameworks/pytorch_geometric_adapter.py's GCNConv
usage), not hand-rolled: GIN's own message-passing update (sum-aggregate
neighbors, an MLP, an epsilon-weighted self term) has no simple closed
form the way GCN's `adjacency @ (features @ W)` does, so there's no
"hand-rolled first, real framework second" story to tell here the way
gcn.py/pytorch_geometric_adapter.py have -- this is the one real
implementation.

**Packing a variable-size graph into this project's fixed-size one-tensor
wire protocol** (this module owns the packing scheme, the same way
architectures/yolov8_seg.py owns `DETECTIONS_NUMEL` for
applications/segmentation.py -- datasets/mutag.py imports `pack_graph`
from here rather than each module defining its own copy): real MUTAG
graphs range from 10-28 nodes, but the wire protocol needs one fixed
shape. Padded to `MAX_NODES=28` (the real dataset maximum, confirmed
directly against `torch_geometric.datasets.TUDataset`) and packed as
three flattened, concatenated blocks -- adjacency `(28, 28)`, node
features `(28, 7)`, and a `(28,)` real-vs-padding mask -- into one
`(1008,)` tensor. `unpack_graph` reverses that per sample, recovers a
dense adjacency, and converts it to `edge_index` via
`torch_geometric.utils.dense_to_sparse` (the same conversion
pytorch_geometric_adapter.py's GCNConv path uses).

Per-sample forward pass (looped across the batch dimension, not a single
batched PyG `Batch` -- batch sizes here are small (this project's
`num_requests`/`batch_size` config, not a real training-scale batch) and
each sample needs its own `edge_index` recovered from its own padded
adjacency slice, so a per-sample loop is simpler and no slower in
practice than building a real PyG `Batch` object just to avoid one):
three `GINConv` layers (each `Linear -> ReLU -> Linear` per PyG's own
`GINConv` convention) over the sample's real `edge_index`, then a
mask-weighted **sum** readout over only the real (non-padding) nodes --
sum pooling, not mean, is GIN's own paper-recommended graph-level readout
(unlike GCN's node-level task, which has no readout step at all) -- then
a final linear classifier to `num_classes` (2 for MUTAG:
mutagenic/non-mutagenic).

Random init, small (hidden_dim=16) -- consistent with every other
architecture here (this is about generating a realistic real-model
network-traffic signature, not real chemistry accuracy, same "traffic
over accuracy" policy as this project's other random-init architectures).
"""
import numpy as np
import torch

from core.registry import ARCHITECTURES

MAX_NODES = 28
NUM_FEATURES = 7
ADJACENCY_NUMEL = MAX_NODES * MAX_NODES
FEATURES_NUMEL = MAX_NODES * NUM_FEATURES
MASK_NUMEL = MAX_NODES
PACKED_NUMEL = ADJACENCY_NUMEL + FEATURES_NUMEL + MASK_NUMEL


def pack_graph(edge_index, x, num_nodes):
    """`edge_index`: (2, E) numpy array. `x`: (num_nodes, NUM_FEATURES)
    numpy array. Pads/truncates to MAX_NODES and flattens to the
    (PACKED_NUMEL,) wire format `unpack_graph` below reverses."""
    num_nodes = min(num_nodes, MAX_NODES)

    adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        if s < MAX_NODES and d < MAX_NODES:
            adjacency[s, d] = 1.0

    features = np.zeros((MAX_NODES, NUM_FEATURES), dtype=np.float32)
    features[:num_nodes] = x[:num_nodes]

    mask = np.zeros((MAX_NODES,), dtype=np.float32)
    mask[:num_nodes] = 1.0

    return np.concatenate([adjacency.reshape(-1), features.reshape(-1), mask])


def unpack_graph(packed_sample):
    """Reverses `pack_graph` on one (PACKED_NUMEL,) torch tensor -> real
    (dense adjacency, node features, real-node mask) torch tensors."""
    adjacency = packed_sample[:ADJACENCY_NUMEL].reshape(MAX_NODES, MAX_NODES)
    features = packed_sample[ADJACENCY_NUMEL:ADJACENCY_NUMEL + FEATURES_NUMEL].reshape(
        MAX_NODES, NUM_FEATURES
    )
    mask = packed_sample[ADJACENCY_NUMEL + FEATURES_NUMEL:]
    return adjacency, features, mask


class _GINClassifier(torch.nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=16, num_classes=2, num_layers=3):
        super().__init__()
        from torch_geometric.nn import GINConv

        self.convs = torch.nn.ModuleList()
        in_dim = num_features
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            in_dim = hidden_dim
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def _forward_one(self, packed_sample):
        from torch_geometric.utils import dense_to_sparse

        adjacency, features, mask = unpack_graph(packed_sample)
        edge_index, _ = dense_to_sparse(adjacency)

        h = features
        for conv in self.convs:
            h = self.relu(conv(h, edge_index))

        # Sum readout over real nodes only -- padding nodes carry zero
        # features and no edges to real nodes, but would still contribute
        # their own bias-driven MLP output to the sum without this mask.
        pooled = (h * mask.unsqueeze(-1)).sum(dim=0)
        return self.classifier(pooled)

    def forward(self, packed_batch):
        if packed_batch.dim() == 1:
            packed_batch = packed_batch.unsqueeze(0)
        return torch.stack([self._forward_one(sample) for sample in packed_batch])


def build(framework_adapter, config):
    return _GINClassifier()


ARCHITECTURES.register(
    "GIN",
    implemented=True,
    family="GNN",
    framework="PyTorch",
    application="Graph Classification",
    num_classes=2,
    input_shape=(PACKED_NUMEL,),
)(build)
