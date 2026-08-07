"""PaddleDetection -- deliberately left a stub after real investigation.

Checked directly: `pip install paddledet` fails while building a
transitive numpy pin from source, with `AttributeError: module 'pkgutil'
has no attribute 'ImpImporter'` inside the legacy `pkg_resources`
compatibility shim -- `pkgutil.ImpImporter` was removed in this Python
version, and the old setuptools/pkg_resources code paddledet's
build chain pulls in still references it. A different specific error
from cv_frameworks/mmdetection_adapter.py's mmcv finding, but the same
root-cause family: legacy packaging code incompatible with a modern
Python/setuptools combination, not a hardware/OS gate.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "PaddleDetection's transitive numpy build fails on a legacy pkg_resources/"
        "pkgutil incompatibility -- see this module's docstring."
    )


CV_FRAMEWORKS.register("PaddleDetection", implemented=False, models="Detection")(build)
