"""FedGraph -- deliberately left a stub after real investigation.

Checked directly: `pip install fedgraph` fails building a transitive
dependency, `torch-scatter` (a PyTorch Geometric-adjacent compiled
extension), with `ModuleNotFoundError: No module named 'torch'` -- even
though torch *is* installed in this environment. Same root cause as
distributed_frameworks/deepspeed_adapter.py's finding: the dependency's
own setup.py imports torch at build time to determine compile flags, and
pip's isolated build environment doesn't expose the already-installed
torch to it by default. `--no-build-isolation` would likely get past this
specific error, but torch-scatter is a real compiled CUDA/CPU extension
requiring its own native build afterward -- getting past this first wall
would very likely just surface a compiler-toolchain wall next (this
machine has none, confirmed repeatedly elsewhere in this file and
distributed_frameworks/). Not attempted further.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedGraph's transitive torch-scatter dependency fails to see the installed "
        "torch during its isolated build -- see this module's docstring."
    )


FL_FRAMEWORKS.register("FedGraph", implemented=False, organization="Rutgers University")(build)
