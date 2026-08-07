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
"""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Spektral is TensorFlow/Keras-based; this project's architecture dispatch "
        "contract (frameworks/pytorch_adapter.py) is PyTorch-only -- see this "
        "module's docstring."
    )


GRAPH_FRAMEWORKS.register("Spektral", implemented=False, application="GNN")(build)
