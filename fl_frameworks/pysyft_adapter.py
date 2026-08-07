"""PySyft -- blocked by a transitive dependency's build, not implemented.
The originally-documented pkg_resources wall is resolved, but each layer
underneath revealed a further, genuine blocker -- confirmed by actually
retrying rather than left as a prediction.

`pip install syft` pulls in an old pinned `pyarrow==17.0.0` with no
prebuilt wheel for this Python, falling back to a source build that
originally failed immediately on `ModuleNotFoundError: No module named
'pkg_resources'`. With the user's explicit authorization, pinned
`setuptools==80.9.0` (the last release still shipping `pkg_resources`) --
see cv_frameworks/mmdetection_adapter.py's docstring for the full
investigation and regression checks confirming this caused no project
regressions.

That gets PySyft's own dependency resolution further: `bcrypt`, `boto3`,
`forbiddenfruit` all now install cleanly, and pyarrow's build script
itself now runs -- but fails one step later needing `Cython` (not
installed). Installed Cython too and retried: pyarrow's *own* legacy
`setup_build.py` then reports a version-detection bug when building this
specific old pin (17.0.0) from an sdist outside a git checkout --
`WARNING: Requested pyarrow==17.0.0 ... but installing version 0.0.0`,
and pip correctly refuses the inconsistent metadata rather than installing
something mislabeled. Building current Arrow C++ from source to get past
this would be a much larger undertaking than every other fix in this
project's compiler/setuptools investigation rounds (a multi-hour native
Arrow build, not a `pip install` with one flag) -- not attempted further,
consistent with this project's standing policy against invasive
multi-step build chains done unprompted.

Same class of blocker as apache-tvm/appfl (build-from-source failure, not
a hardware/OS gate) but one level removed: not this package's own build
system, a transitive dependency's, and now two dependencies removed
(pyarrow's own build tooling, not even PySyft's). Revisit once PySyft
updates its pyarrow pin to a version with a prebuilt wheel for this
Python, or once pyarrow's legacy sdist version-detection is fixed upstream.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "PySyft's pinned pyarrow==17.0.0 dependency's own legacy build script "
        "reports inconsistent version metadata when building from source on this "
        "Python -- see this module's docstring."
    )


FL_FRAMEWORKS.register("PySyft", implemented=False, organization="OpenMined")(build)
