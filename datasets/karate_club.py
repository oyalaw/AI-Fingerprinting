"""Karate Club -- Zachary's Karate Club graph (Zachary, 1977), the
standard toy dataset for GCN node classification (it's the dataset Kipf &
Welling's original GCN paper demonstrates on). Not one of this project's
original 7 datasets -- added because GCN's natural task (node
classification on a real graph) doesn't map onto any of them, the same
kind of well-justified addition as ONNX Runtime Mobile/MediaPipe on the
frameworks side.

Loaded via `networkx.karate_club_graph()` -- confirmed directly rather
than hand-transcribed: 34 nodes, 78 edges, nodes labeled by which of the
two factions ("Mr. Hi" vs "Officer") they ended up joining after the
club's real-world split, a 17/17 class balance.

GCN is transductive: it classifies every node in one forward pass over
the whole graph, there's no such thing as classifying one node in
isolation from the graph structure. So unlike every other dataset here,
`samples(n)` doesn't yield n independent examples -- there's only one
graph -- it yields the same graph-wide input n times, each paired with
the full 34-node label vector. That's a legitimate real modeling choice
for transductive graph learning, not a workaround, and it's also a
realistic serving pattern for a traffic-fingerprinting testbed: repeated
inference requests against a static graph (e.g. a recommendation system
re-scoring the same social graph) is exactly the kind of workload this
project exists to fingerprint.

The (2, 34, 34) stacked array packs both graph structure and node
features into the single tensor this project's one-tensor wire protocol
needs: channel 0 is the symmetrically-normalized adjacency (via
families/gnn's shared helper), channel 1 is an identity matrix used as
node features (the graph has no intrinsic node features -- one-hot node
IDs is the standard featureless-graph convention, used in the original
Kipf & Welling reference implementation).
"""
import numpy as np

from core.registry import DATASETS
from datasets.base import Dataset
from families.gnn import normalize_adjacency


class KarateClubDataset(Dataset):
    def __init__(self):
        import networkx as nx

        graph = nx.karate_club_graph()
        num_nodes = graph.number_of_nodes()

        adjacency = nx.to_numpy_array(graph, dtype=np.float32)
        normalized = normalize_adjacency(adjacency)
        identity_features = np.eye(num_nodes, dtype=np.float32)
        self._stacked_input = np.stack([normalized, identity_features])  # (2, 34, 34)

        self._labels = [graph.nodes[n]["club"] for n in graph.nodes]

    def samples(self, n):
        for _ in range(n):
            yield self._stacked_input, self._labels


DATASETS.register("Karate Club", implemented=True)(KarateClubDataset)
