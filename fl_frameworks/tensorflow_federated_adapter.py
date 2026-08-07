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
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "TensorFlow Federated needs TensorFlow, which has no wheel for this Python "
        "version -- see this module's docstring."
    )


FL_FRAMEWORKS.register("TensorFlow Federated", implemented=False, organization="Google")(build)
