"""TensorFlow Federated -- deliberately left a stub, for two independent
real reasons, checked directly rather than assumed.

First wall hit: `pip install tensorflow-federated` fails building a
transitive `grpcio` pin from source, with `ModuleNotFoundError: No module
named 'pkg_resources'` -- the same legacy-setuptools-API-unavailable-in-
isolated-builds root cause as cv_frameworks/mmdetection_adapter.py's mmcv
finding and fl_frameworks/pysyft_adapter.py's pyarrow finding.

Even past that, TFF is fundamentally built on TensorFlow, and plain
`tensorflow` has no wheel at all for this Python version -- the same
root-cause blocker frameworks/tensorflow_adapter.py's own stub already
documents, and the same wall fl_frameworks/fedscale_adapter.py's
Aggregator import chain terminates at. This is the third stub blocked on
that specific wall (alongside graph_frameworks/spektral_adapter.py's
framework-mismatch finding, which is a different kind of TF-related
block). Revisit all of these together once TensorFlow itself has a real
install here.

That second wall is now resolved (see frameworks/tensorflow_adapter.py's
and fl_frameworks/fedscale_adapter.py's docstrings -- TensorFlow 2.21.0
ships a real `cp313` wheel and installs cleanly on Ubuntu). Re-checked
`pip install tensorflow-federated` directly with that in place: the
*first* wall is still there, unchanged, and it's what actually blocks
this now. Confirmed why it's not fixed by TensorFlow being available or
by moving to Linux: `tensorflow-federated`'s own pin is
`grpcio~=1.46.3` (from ~2022) -- checked directly via grpcio's own PyPI
file index, that release has zero `cp312`/`cp313` wheels on *any*
platform, so it always falls back to a source build, which hits the same
`pkg_resources`-removed-from-modern-setuptools wall regardless of OS.
Same Python-version-ceiling class of finding as fl_frameworks/
pysyft_adapter.py's pyarrow pin and cv_frameworks/paddledetection_adapter.py's
numpy pin -- not something switching platforms changes. Revisit once TFF
relaxes its grpcio pin to a version with a current wheel.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "TensorFlow Federated needs TensorFlow, which has no wheel for this Python "
        "version -- see this module's docstring."
    )


FL_FRAMEWORKS.register("TensorFlow Federated", implemented=False, organization="Google")(build)
