"""FedML -- deliberately left a stub after real investigation. Got further
after installing a real C/C++ compiler (Visual Studio Build Tools, installed
specifically to unblock this and several other stubs sharing the "no
compiler" root cause) -- the original numpy-build blocker is gone, but a
different, unrelated one replaced it.

`pip install fedml` no longer fails on numpy (that finding is resolved).
It now fails building a transitive dependency instead: `pathtools` (pulled
in by `wandb`, one of FedML's own dependencies), whose `setup.py` does
`import imp` -- the `imp` module was removed outright in Python 3.12+
(deprecated since 3.4, finally deleted). This is 2010s-era legacy code that
was never updated for modern Python, not a missing-compiler problem the way
the numpy build was -- a compiler can't fix a module that no longer exists
in the standard library. `pathtools` itself has been unmaintained for years
(its functionality was folded into `watchdog`), so this is really "FedML
depends on wandb, which still depends on an abandoned package." Revisit if
FedML/wandb drop the pathtools dependency, or if running on an older Python
becomes acceptable for this project.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedML's transitive pathtools dependency (via wandb) does `import imp`, "
        "removed in Python 3.12+ -- see this module's docstring."
    )


FL_FRAMEWORKS.register("FedML", implemented=False, organization="FedML Inc.")(build)
