"""MMSegmentation -- deliberately left a stub, same root cause as
cv_frameworks/mmdetection_adapter.py: it's part of the OpenMMLab family
and depends on mmcv, whose own setup.py fails on a legacy pkg_resources
API unavailable in pip's modern isolated build environment (confirmed
directly against mmcv itself, not assumed). See that module's docstring
for the exact error and cv_frameworks/openmmlab_adapter.py for the third
member of this family hitting the identical wall.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "MMSegmentation depends on mmcv, whose own build fails on a legacy "
        "pkg_resources API -- see cv_frameworks/mmdetection_adapter.py's docstring."
    )


CV_FRAMEWORKS.register("MMSegmentation", implemented=False, models="Segmentation")(build)
