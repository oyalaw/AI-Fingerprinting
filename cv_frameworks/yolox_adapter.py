"""YOLOX -- deliberately left a stub after real investigation.

Checked directly: `pip install yolox` fails while building a transitive
dependency, not yolox itself -- its setup.py pulls in a pinned `onnx`
version with no prebuilt wheel for this Python, which falls back to
building from source and immediately fails with
`AssertionError: Could not find "cmake" executable!`. Same class of
blocker as apache-tvm (frameworks/tvm_adapter.py) and APPFL
(fl_frameworks/appfl_adapter.py): a missing build toolchain, not a
hardware/OS gate. Installing cmake just to unblock one transitive
dependency was judged the same kind of invasive fix as installing a C
compiler for those -- not done unprompted.

Also worth noting for whoever revisits this: even once installed, YOLOX
would need to be a new architecture entry (like architectures/yolov8_seg.py
was for its checkpoint), not a cv_framework dispatch under the existing
"YOLOv8" architecture -- YOLOX is a genuinely different detector design
(anchor-free, decoupled head), not an alternate implementation of YOLOv8
the way graph_frameworks/pytorch_geometric_adapter.py is an alternate
GCN implementation.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "YOLOX's own install fails on a transitive dependency (onnx) needing "
        "cmake to build from source -- see this module's docstring."
    )


CV_FRAMEWORKS.register("YOLOX", implemented=False, models="Detection")(build)
