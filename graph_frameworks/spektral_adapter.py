"""Spektral -- deliberately left a stub, not a gap: a real architectural
mismatch, not a missing wheel.

`spektral` installs fine (checked: 1.3.1 has a current PyPI wheel). The
blocker is deeper -- Spektral is built on Keras/TensorFlow, not PyTorch.
Every architecture in this project (including architectures/gcn.py's
default and graph_frameworks/pytorch_geometric_adapter.py's dispatch)
returns a PyTorch nn.Module, because frameworks/pytorch_adapter.py's
predict()/serialize()/deserialize() -- the only path roles/server.py
calls -- all operate on torch.Tensor. A Spektral model would be a
tf.keras.Model with numpy/tf.Tensor I/O, incompatible with that contract
without either a second, TensorFlow-backed framework adapter (frameworks/
tensorflow_adapter.py is itself still a stub) or a torch<->tf conversion
shim neither this project nor Spektral provides.

That's a different, larger kind of gap than DGL (wrong version) or
StellarGraph (no wheel) -- revisit once frameworks/tensorflow_adapter.py
itself is real, not before.

That revisit condition is technically met (frameworks/tensorflow_adapter.py
is now `implemented=True` and genuinely installs/runs -- see that
module's docstring), but re-checking confirms it doesn't actually change
anything here, for a precise reason worth recording rather than assuming:
`frameworks/tensorflow_adapter.py` converts an existing *PyTorch*
architecture to TensorFlow for inference (via `onnx2tf`) -- it doesn't
give a `graph_framework: Spektral` dispatch (the shape
`graph_frameworks/pytorch_geometric_adapter.py` uses for GCN) anywhere to
land, because that dispatch would need to construct a *native* Spektral
model, and that's exactly what's structurally incompatible. Confirmed
directly on Ubuntu with TensorFlow actually available this time: `pip
install spektral` succeeds cleanly, `import spektral` works, and
`spektral.layers.GCNConv` is real -- but a real `tf.keras.Sequential`
model built alongside it has neither `.state_dict()` nor `.parameters()`
(confirmed directly, both `hasattr(...)` checks are `False`), the exact
two calls `frameworks/pytorch_adapter.py` and every FL/distributed
adapter's training loop make unconditionally. Unlike
graph_frameworks/pytorch_geometric_adapter.py's GCNConv (a PyTorch layer,
so `build_gcn()` can still return a real `torch.nn.Module`), there's no
way for a `graph_framework: Spektral` dispatch to return something
`architecture_entry.build()`'s callers can use without either a second,
TensorFlow-backed framework adapter with its own predict()/serialize()
path (not what frameworks/tensorflow_adapter.py is) or a real torch<->tf
model-conversion shim, neither of which exist. Stays a stub; the earlier
"revisit once TensorFlow is real" framing was imprecise about which real
TensorFlow the fix actually needed.
"""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Spektral is TensorFlow/Keras-based; this project's architecture dispatch "
        "contract (frameworks/pytorch_adapter.py) is PyTorch-only -- see this "
        "module's docstring."
    )


GRAPH_FRAMEWORKS.register("Spektral", implemented=False, application="GNN")(build)
