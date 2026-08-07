"""MMDetection -- deliberately left a stub after real investigation. The
originally-documented pkg_resources wall is resolved, but mmcv (the
dependency mmdet's actual detection ops live in) hits a different, deeper
wall underneath: a genuine CPython 3.14 behavior change, not a packaging
gap.

`pip install mmdet` succeeds cleanly (pure Python wheel, 3.3.0), but
`import mmdet` fails immediately with `ModuleNotFoundError: No module
named 'mmcv'` -- mmdet's actual detection models (NMS, ROI Align, etc.)
depend entirely on mmcv's compiled ops, not bundled or auto-installed as a
dependency. Installing mmcv directly originally failed on
`ModuleNotFoundError: No module named 'pkg_resources'` (this environment's
modern setuptools no longer bundles it). With the user's explicit
authorization, pinned `setuptools==80.9.0` (the last release still
shipping `pkg_resources`) -- confirmed no regressions from this change via
a full registry list and a real ResNet18 roundtrip, and confirmed
`speech_frameworks/espnet_adapter.py`'s real dispatch still works
unchanged.

That resolves the pkg_resources import, but `pip install --no-build-isolation
mmcv` now fails differently and further along: `KeyError: '__version__'`
inside mmcv's own `setup.py`'s `get_version()`, which does
`exec(compile(open('mmcv/version.py').read(), ...)); return
locals()['__version__']`. Reproduced this exact failure directly with a
minimal standalone snippet: as of Python 3.13's PEP 667 ("Consistent views
of namespaces"), `exec()` running inside an optimized function scope no
longer writes its assigned names into that function's `locals()` -- a real
CPython language-semantics change, not a mmcv bug or a version-detection
edge case. mmcv's `setup.py` predates this change and has no other way to
read its own version string. Same failure signature (and now understood
root cause) as fl_frameworks/easyfl_adapter.py's `KeyError: '__version__'`
finding. Revisit once mmcv updates this pattern (e.g. reading the version
via regex/AST instead of exec), or if this project ever runs on an older
Python. This blocks the whole OpenMMLab family sharing mmcv -- see
cv_frameworks/mmsegmentation_adapter.py and openmmlab_adapter.py, which
hit the identical wall.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "MMDetection depends on mmcv, whose own setup.py hits a real Python 3.13+ "
        "exec()/locals() behavior change while reading its version string -- see "
        "this module's docstring."
    )


CV_FRAMEWORKS.register("MMDetection", implemented=False, models="Detection models")(build)
