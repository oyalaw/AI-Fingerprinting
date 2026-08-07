"""OpenMMLab -- deliberately left a stub, same root cause as
cv_frameworks/mmdetection_adapter.py and mmsegmentation_adapter.py: the
whole family (this entry's own `openmim` installer included) depends on
mmcv, whose own setup.py fails on a legacy pkg_resources API unavailable
in pip's modern isolated build environment (confirmed directly against
mmcv itself). See mmdetection_adapter.py's docstring for the exact error.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "OpenMMLab's toolchain depends on mmcv, whose own build fails on a legacy "
        "pkg_resources API -- see cv_frameworks/mmdetection_adapter.py's docstring."
    )


CV_FRAMEWORKS.register("OpenMMLab", implemented=False, models="Vision toolkit")(build)
