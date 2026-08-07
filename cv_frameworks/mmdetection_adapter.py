"""MMDetection -- deliberately left a stub after real investigation.

Checked directly: `pip install mmdet` succeeds cleanly (pure Python wheel,
3.3.0), but `import mmdet` fails immediately with `ModuleNotFoundError:
No module named 'mmcv'` -- mmdet's actual detection models (NMS, ROI
Align, etc.) depend entirely on mmcv's compiled ops, not bundled or
auto-installed as a dependency. Tried installing mmcv directly: its own
setup.py fails at the build-requirements step with `ModuleNotFoundError:
No module named 'pkg_resources'` -- mmcv's build script uses the legacy
setuptools pkg_resources API, unavailable in pip's modern isolated build
environment. Same root-cause class as fl_frameworks/pysyft_adapter.py's
pyarrow finding. This blocks the whole OpenMMLab family sharing mmcv --
see cv_frameworks/mmsegmentation_adapter.py and openmmlab_adapter.py,
which hit the identical wall.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "MMDetection depends on mmcv, whose own build fails on a legacy "
        "pkg_resources API unavailable in modern isolated builds -- see this "
        "module's docstring."
    )


CV_FRAMEWORKS.register("MMDetection", implemented=False, models="Detection models")(build)
