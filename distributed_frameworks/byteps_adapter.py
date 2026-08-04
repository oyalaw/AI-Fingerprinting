"""BytePS -- registered but not yet implemented.

Checked directly: `pip install byteps` fails while building its bundled
`ps-lite` C++ library -- its Makefile-based build invokes `make`, which
doesn't exist on Windows (`'make' is not recognized as an internal or
external command`). Would need a Unix build toolchain (WSL, MSYS2/MinGW
with make on PATH, or a Linux machine/container) to get past this step at
all, and likely further native-networking-library assumptions (RDMA/UCX
were both already reported disabled in the same build log) beyond it.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("BytePS is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("BytePS", implemented=False, organization="Bytedance")(build)
