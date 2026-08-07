"""OpenMMLab -- deliberately left a stub, same root cause as
cv_frameworks/mmdetection_adapter.py and mmsegmentation_adapter.py: the
whole family (this entry's own `openmim` installer included) depends on
mmcv. The originally-documented pkg_resources wall is resolved (setuptools
pinned to 80.9.0 with the user's explicit authorization -- see
mmdetection_adapter.py's docstring), but mmcv's own setup.py hits a
different, deeper wall underneath: a genuine CPython 3.13+ behavior change
(PEP 667) breaking its `get_version()` helper's `exec()`/`locals()`
pattern, confirmed directly with a minimal standalone repro. See
mmdetection_adapter.py's docstring for the exact error.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "OpenMMLab's toolchain depends on mmcv, whose own setup.py hits a real "
        "Python 3.13+ exec()/locals() behavior change -- see "
        "cv_frameworks/mmdetection_adapter.py's docstring."
    )


CV_FRAMEWORKS.register("OpenMMLab", implemented=False, models="Vision toolkit")(build)
