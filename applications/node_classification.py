"""Node Classification application -- not one of this project's original
8 applications, added for the same reason datasets/karate_club.py was:
GCN's natural task doesn't map onto any of them. See that module's
docstring for the full justification.

preprocess: the (2, 34, 34) stacked adjacency+features array
datasets/karate_club.py produces -> a plain float tensor (no per-image/
per-token normalization needed, the graph module already built a fully
normalized array).

postprocess: GCN classifies every node in the graph in one forward pass
(transductive, see karate_club.py's docstring), so the model output is
(34, num_classes) logits, not a single prediction -- postprocess returns
a list of one predicted label per node, not a single value like every
other application here. That's a real difference in what "one result"
means for a transductive graph task, not an inconsistency.
"""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_FACTION_LABELS = ("Mr. Hi", "Officer")


class NodeClassification(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return torch.from_numpy(raw_sample.copy()).float()
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        if output_tensor.dim() == 3:
            output_tensor = output_tensor.squeeze(0)
        predicted_indices = torch.argmax(output_tensor, dim=-1).tolist()
        return [_FACTION_LABELS[idx] for idx in predicted_indices]


APPLICATIONS.register("Node Classification", implemented=True)(NodeClassification)
