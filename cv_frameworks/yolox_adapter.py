"""YOLOX -- deliberately left a stub after real investigation. Got further
after installing a real C/C++ compiler (Visual Studio Build Tools, installed
specifically to unblock this and several other stubs sharing the "no
compiler" root cause) -- the original cmake/onnx blocker is gone, but a
genuine bug in YOLOX's own packaging replaced it.

`pip install yolox` no longer fails on the transitive onnx/cmake build
(that finding is resolved: a compiler was the missing piece there). Retried
with `--no-build-isolation` after first hitting a
`ModuleNotFoundError: No module named 'torch'` inside the isolated build
sandbox (pip's build isolation doesn't see the already-installed torch in
this project's own environment -- same class of build-isolation blindness
documented elsewhere in this project). With that worked around, install now
fails inside YOLOX's own `setup.py`:
`FileNotFoundError: [Errno 2] No such file or directory: 'requirements.txt'`
-- its `setup.py` reads `requirements.txt` via a path that only exists in a
git checkout of the YOLOX repo, not inside the isolated temp directory pip
builds an sdist in. This is a genuine packaging bug in YOLOX's own release
artifact (its sdist doesn't bundle a file its own setup.py requires), not
something a compiler or `--no-build-isolation` can route around. Revisit if
YOLOX ships a fixed sdist, or if installing directly from a git checkout
(`pip install git+https://github.com/Megvii-BaseDetection/YOLOX`) becomes
worth doing instead of a PyPI install.

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
        "YOLOX's own sdist is missing a requirements.txt its setup.py requires -- "
        "a packaging bug in the release artifact, see this module's docstring."
    )


CV_FRAMEWORKS.register("YOLOX", implemented=False, models="Detection")(build)
