"""Detectron2 -- deliberately left a stub, not a gap: never published to
PyPI at all, by design.

Checked directly: `pip install detectron2` returns "No matching
distribution found". Consistent with Detectron2's real, well-documented
install story: Meta only ever distributes it via
`pip install git+https://github.com/facebookresearch/detectron2.git` (a
source build compiling custom CUDA/C++ ops), or prebuilt wheels hosted on
Meta's own index for specific pinned torch+CUDA+Python combinations, none
of which include this Python version. Installing from the git URL would
still hit a compiler-toolchain wall -- this machine has none, confirmed
repeatedly elsewhere in this file (cv_frameworks/mmdetection_adapter.py,
distributed_frameworks/horovod_adapter.py, etc.).
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Detectron2 has no PyPI package and its source build needs a C++/CUDA "
        "compiler toolchain this machine doesn't have -- see this module's docstring."
    )


CV_FRAMEWORKS.register("Detectron2", implemented=False, models="Meta vision models")(build)
