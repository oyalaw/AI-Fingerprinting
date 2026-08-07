"""EasyFL -- deliberately left a stub after real investigation.

Checked directly: `pip install easyfl` fails with `KeyError: '__version__'`
inside the package's own setup.py, in its `get_version()` helper. Root
cause now confirmed precisely (previously just described as "a bug"):
`get_version()` does `exec(compile(open(version_file).read(), ...));
return locals()['__version__']` -- and reproduced directly with a minimal
standalone snippet that this fails on Python 3.13+ specifically, because
of PEP 667 ("Consistent views of namespaces"): `exec()` running inside an
optimized function scope no longer writes its assigned names into that
function's `locals()`. Not unique to EasyFL -- the identical pattern and
failure signature was found independently in mmcv's own `setup.py` (see
cv_frameworks/mmdetection_adapter.py's docstring), so this is a broader
"old setup.py idiom, new Python" incompatibility class, not a one-off
EasyFL bug. Consistent with EasyFL having had no PyPI release since 2022
-- likely abandoned, so unlikely to ever adopt a fixed pattern itself.
Revisit only if a fixed release appears, or if this project ever runs on
an older Python.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "EasyFL's own setup.py hits a real Python 3.13+ exec()/locals() behavior "
        "change while reading its version string -- see this module's docstring."
    )


FL_FRAMEWORKS.register("EasyFL", implemented=False, organization="Microsoft")(build)
