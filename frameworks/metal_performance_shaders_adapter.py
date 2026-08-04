"""Metal Performance Shaders -- deliberately left a stub, not a gap.

Unlike every other stub in this registry, this isn't blocked by a missing
wheel, missing hardware, or an untrusted bundled binary -- it's blocked by
there being no real, distinct thing to implement. Apple has never shipped
official Python bindings for MPSGraph/Metal Performance Shaders; the only
Python-reachable paths onto Metal are:

  - PyTorch's own `mps` device (`torch.device("mps")`) -- already
    implemented as its own registry entry, frameworks/mps_backend_adapter.py.
  - coremltools' CoreML conversion, which can select Metal-backed compute
    units at prediction time -- already implemented, frameworks/coreml_adapter.py.

Writing a third "Metal Performance Shaders" adapter would necessarily
just be one of those two again under a different label -- inflating the
implemented count with duplicate work rather than adding anything real.
If Apple ever ships a standalone Python MPSGraph API, or there's a
genuine use case distinct from the two paths above, revisit this;
until then it stays a stub on principle, not by omission.
"""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Metal Performance Shaders has no distinct Python API -- see this "
        "module's docstring. Use MPS Backend (PyTorch) or CoreML instead."
    )


FRAMEWORKS.register(
    "Metal Performance Shaders", implemented=False, organization="Apple", platforms=["ios", "ipados", "macos"]
)(build)
