"""PySyft -- blocked by a transitive dependency's build, not implemented.

Checked directly: `pip install syft` pulls in an old pinned `pyarrow`
version with no prebuilt wheel for this Python, and its build falls back
to source -- which fails immediately with `ModuleNotFoundError: No module
named 'pkg_resources'` inside pyarrow's own legacy setup.py, which isn't
available in the isolated build environment on this Python version.

Same class of blocker as apache-tvm/appfl (build-from-source failure, not
a hardware/OS gate) but one level removed: not this package's own build
system, a transitive dependency's. Forcing a workaround (patching the
build env, pinning setuptools down to keep pkg_resources) was judged the
same kind of invasive fix as installing a C compiler for apache-tvm --
not done unprompted. Revisit once PySyft updates its pyarrow pin.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "PySyft's pinned pyarrow dependency fails to build on this Python version -- see this module's docstring."
    )


FL_FRAMEWORKS.register("PySyft", implemented=False, organization="OpenMined")(build)
