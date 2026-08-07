"""MMSegmentation -- deliberately left a stub, same root cause as
cv_frameworks/mmdetection_adapter.py: it's part of the OpenMMLab family
and depends on mmcv. The originally-documented pkg_resources wall is
resolved (setuptools pinned to 80.9.0, the last release still shipping
pkg_resources, with the user's explicit authorization -- see
mmdetection_adapter.py's docstring for the full investigation and
regression checks), but mmcv's own setup.py hits a different, deeper wall
underneath: a genuine CPython 3.13+ behavior change (PEP 667) where
`exec()` inside an optimized function scope no longer writes into that
function's `locals()`, breaking mmcv's `get_version()` helper. Confirmed
directly with a minimal standalone repro, not assumed. See
mmdetection_adapter.py's docstring for the exact error and
cv_frameworks/openmmlab_adapter.py for the third member of this family
hitting the identical wall.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "MMSegmentation depends on mmcv, whose own setup.py hits a real Python "
        "3.13+ exec()/locals() behavior change -- see "
        "cv_frameworks/mmdetection_adapter.py's docstring."
    )


CV_FRAMEWORKS.register("MMSegmentation", implemented=False, models="Segmentation")(build)
