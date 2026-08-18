"""Graph Classification application -- not one of this project's original
8 applications, added alongside architectures/gin.py and
datasets/mutag.py: GIN's natural task (classify a whole graph, not a
node) doesn't map onto Node Classification (transductive, one graph,
per-node labels) or any of the other 8. See datasets/mutag.py's docstring
for the full justification.

preprocess: the `(1008,)` packed adjacency+features+mask array
datasets/mutag.py's `pack_graph` produces -> a plain float tensor (the
packing already did all the real work; no per-sample normalization
needed here, same as Node Classification's preprocess).

postprocess: one graph in, one scalar mutagenic/non-mutagenic label out
-- the same "real per-sample classification" shape as Sentiment
Analysis/Image Classification (unlike Node Classification's per-node
list), since graph classification genuinely is sample-shaped.
"""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_LABELS = ("non-mutagenic", "mutagenic")


class GraphClassification(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return torch.from_numpy(raw_sample.copy()).float()
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        if output_tensor.dim() == 2:
            output_tensor = output_tensor.squeeze(0)
        probs = torch.softmax(output_tensor, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        return _LABELS[idx]


APPLICATIONS.register("Graph Classification", implemented=True, datasets=["MUTAG"])(
    GraphClassification
)
