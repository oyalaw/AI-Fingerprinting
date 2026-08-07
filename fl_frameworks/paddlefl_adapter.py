"""PaddleFL -- deliberately left a stub, not a gap: never published as a
pip package at all.

Checked directly: `pip install paddlefl` returns "No matching
distribution found" -- no wheel or sdist at all. Consistent with PaddleFL
having had no real maintenance activity in years; Baidu's federated
learning effort moved to other projects. Even setting that aside,
PaddleFL is built on PaddlePaddle, not PyTorch -- the same framework-
family mismatch documented on graph_frameworks/spektral_adapter.py, so
this would need its own non-PyTorch model path even if it were
installable.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "PaddleFL has no pip package -- see this module's docstring."
    )


FL_FRAMEWORKS.register("PaddleFL", implemented=False, organization="Baidu")(build)
