"""EasyFL -- deliberately left a stub after real investigation.

Checked directly: `pip install easyfl` fails with `KeyError: '__version__'`
inside the package's own setup.py, in its `get_version()` helper -- a bug
in EasyFL's own packaging (its version-parsing logic doesn't find what it
expects, unrelated to anything about this environment/platform/Python
version). A different category of finding than the build-toolchain
blockers elsewhere in this file (FedML, APPFL): there's no missing
compiler or wheel to wait for here, the package's own build script is
broken as published. Consistent with EasyFL having had no PyPI release
since 2022 -- likely abandoned. Revisit only if a fixed release appears.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "EasyFL's own setup.py fails with KeyError: '__version__' -- a bug in its "
        "packaging, not an environment issue -- see this module's docstring."
    )


FL_FRAMEWORKS.register("EasyFL", implemented=False, organization="Microsoft")(build)
