"""FedML -- deliberately left a stub after real investigation.

Checked directly: `pip install fedml` fails while building a transitive
numpy pin from source (meson build backend probes for a C/C++ compiler --
icl/cl/cc/gcc/clang/clang-cl/pgcc all report "WinError 2: The system
cannot find the file specified") -- this machine has no compiler
installed. Same class of blocker as fl_frameworks/appfl_adapter.py's
numpy finding and cv_frameworks/yolox_adapter.py's onnx finding:
installing a build toolchain just to get one transitive dependency's
wheel was judged too invasive to do unprompted. Revisit if a compatible
numpy wheel becomes available, or a compiler is installed for other
reasons.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedML's transitive numpy pin fails to build from source (no compiler) -- "
        "see this module's docstring."
    )


FL_FRAMEWORKS.register("FedML", implemented=False, organization="FedML Inc.")(build)
