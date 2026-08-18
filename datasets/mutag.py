"""MUTAG -- the standard graph-classification benchmark (Debnath et al.,
1991; the reference dataset most GNN graph-classification papers,
including FedGraph's own, use as their first example). Not one of this
project's original datasets -- added alongside architectures/gin.py and
applications/graph_classification.py for the same reason
datasets/karate_club.py was added for GCN: this task doesn't map onto
any existing dataset/application here.

Loaded via `torch_geometric.datasets.TUDataset(root, name="MUTAG")` --
confirmed directly rather than assumed: 188 real molecular graphs (nitro
compounds), each labeled mutagenic/non-mutagenic against Salmonella
typhimurium, 7 node features (one-hot atom type), real graph sizes from
10 to 28 nodes.

Unlike Karate Club (one graph, classified transductively), MUTAG is
genuinely sample-shaped -- each graph is one independent classification
example, the same "many independent samples, one scalar label each"
shape every other dataset here has, just with a graph instead of an
image/token sequence as the per-sample input. `samples(n)` cycles through
the real 188 graphs (not synthetic), same pattern as
datasets/imdb.py/sst2.py cycling through their real text samples.

Packing each variable-size graph into this project's fixed-size
one-tensor wire protocol is owned by architectures/gin.py (`pack_graph`/
`MAX_NODES`/`NUM_FEATURES`, imported directly here so the two files can't
drift out of sync -- the same discipline applications/segmentation.py's
import of `DETECTIONS_NUMEL` from architectures/yolov8_seg.py follows),
not duplicated in this module.
"""
from architectures.gin import NUM_FEATURES, pack_graph
from core.registry import DATASETS
from datasets.base import Dataset

_LABELS = ("non-mutagenic", "mutagenic")


class MUTAGDataset(Dataset):
    def __init__(self, root="./data/mutag"):
        from torch_geometric.datasets import TUDataset

        torch_dataset = TUDataset(root=root, name="MUTAG")
        if torch_dataset.num_node_features != NUM_FEATURES:
            raise ValueError(
                f"MUTAG's real node-feature dimension ({torch_dataset.num_node_features}) "
                f"no longer matches architectures/gin.py's NUM_FEATURES={NUM_FEATURES} -- "
                f"TUDataset's format must have changed; update both together."
            )

        self._packed = []
        self._labels = []
        for graph in torch_dataset:
            packed = pack_graph(graph.edge_index.numpy(), graph.x.numpy(), graph.num_nodes)
            self._packed.append(packed)
            self._labels.append(_LABELS[int(graph.y.item())])

    def samples(self, n):
        for i in range(n):
            idx = i % len(self._packed)
            yield self._packed[idx], self._labels[idx]


DATASETS.register("MUTAG", implemented=True)(MUTAGDataset)
