"""APPFL -- registered but not yet implemented.

Checked directly: `pip install appfl` pulls in a numpy version with no
prebuilt wheel for this Python/platform combo, so pip falls back to
building numpy from source via meson, which then fails immediately --
this machine has no C compiler (`clang-cl`/`pgcc` both report
"WinError 2: The system cannot find the file specified"). Same class of
blocker as apache-tvm (frameworks/tvm_adapter.py): installing a full
build toolchain just to get one dependency's wheel was judged too
invasive to do unprompted. Not a hardware/OS gate like TensorRT/CoreML --
revisit if a compatible numpy wheel becomes available, or a build
toolchain is added to this environment.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("APPFL is not yet implemented -- see module docstring.")


FL_FRAMEWORKS.register("APPFL", implemented=False, organization="Argonne National Laboratory")(build)
