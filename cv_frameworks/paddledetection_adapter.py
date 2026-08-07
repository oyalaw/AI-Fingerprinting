"""PaddleDetection -- deliberately left a stub after real investigation.
The originally-documented pkgutil/pkg_resources wall no longer reproduces
after pinning setuptools to 80.9.0 (see
cv_frameworks/mmdetection_adapter.py's docstring for that change and its
regression checks) -- but retrying reveals a different, deeper, equally
unfixable-here wall in the same transitive numpy pin.

`pip install --no-build-isolation paddledet` now gets further: its
transitive `numpy==1.23.5` pin's build script runs past the original
`pkgutil.ImpImporter` error entirely. It fails one step later instead,
inside `numpy.distutils` (numpy's own legacy pre-`meson` build system):
`ModuleNotFoundError: No module named 'distutils.msvccompiler'` --
`distutils` itself was removed outright from the Python standard library
in Python 3.12 (deprecated since 3.10, removed per PEP 632), and this old
numpy release's Windows compiler-detection code still imports it directly.
Not a missing package pip could install -- there is no `distutils` to add
back, it's gone from the interpreter itself. A permanent, version-pin-side
incompatibility: this specific ancient numpy release cannot build on any
Python >= 3.12, regardless of setuptools version or compiler availability.
Same root-cause family as the original finding (legacy packaging code
incompatible with a modern Python), but a different, harder, standard-
library-level wall underneath it, not a hardware/OS gate. Revisit only if
PaddleDetection relaxes its numpy pin to a release with a real
Python-3.12+-compatible build system.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "PaddleDetection's transitive numpy==1.23.5 pin needs the stdlib "
        "distutils.msvccompiler module, removed outright in Python 3.12+ -- see "
        "this module's docstring."
    )


CV_FRAMEWORKS.register("PaddleDetection", implemented=False, models="Detection")(build)
