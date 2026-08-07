"""Detectron2 -- real, but informational, the same "nothing to dispatch
to" situation as cv_frameworks/ultralytics_adapter.py documents for
YOLOv8: architectures/detectron2.py builds and runs Detectron2's real
`GeneralizedRCNN` directly and unconditionally, so selecting
`framework=PyTorch, architecture=Detectron2` already means "use
Detectron2," whether or not `cv_framework=Detectron2` is separately set.
This entry documents that fact and is registered `implemented` for that
reason, not because setting it changes anything at runtime.

Previously a stub: never published to PyPI, and installing from Meta's
documented git-source path would have hit a missing-compiler wall on this
machine. Both resolved -- see architectures/detectron2.py's docstring for
the full investigation: a real C/C++ compiler (Visual Studio Build Tools)
was installed to unblock several other stubs sharing that root cause, and
`pip install git+https://github.com/facebookresearch/detectron2.git` now
succeeds, building Detectron2's custom ops as a CPU-only extension (no
CUDA toolkit on this machine, and Detectron2's own setup.py falls back to
CPU-only compilation rather than failing when none is found).
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    return "detectron2"


CV_FRAMEWORKS.register("Detectron2", implemented=True, models="Meta vision models")(build)
