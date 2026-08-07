"""StellarGraph -- deliberately left a stub after real investigation.

Checked directly: `pip install stellargraph` fails outright with
"No matching distribution found" -- no wheel or sdist at all on PyPI for
this Python version. Not a version-mismatch or build-from-source
situation like DGL/apache-tvm/APPFL; there's simply nothing to install.

That's consistent with the project's own status: StellarGraph's
maintainers archived the repository in 2021 ("This library is not
currently maintained and its dependencies have become incompatible with
the latest Python releases") and stopped publishing new wheels. Revisit
only if the project is ever revived or an older Python version becomes
relevant.
"""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "StellarGraph has no installable distribution for this Python version -- "
        "see this module's docstring (project archived by its maintainers in 2021)."
    )


GRAPH_FRAMEWORKS.register("StellarGraph", implemented=False, application="GNN")(build)
