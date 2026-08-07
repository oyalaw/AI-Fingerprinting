"""FedGraph -- deliberately left a stub after real investigation. The
originally-predicted chain (fix build-isolation, then hit a
compiler-toolchain wall) is now confirmed only halfway -- a real compiler
resolved that layer cleanly, but a completely different, unrelated
blocker was underneath it instead of the predicted one.

`pip install --no-build-isolation fedgraph` (with a real C/C++ compiler
now installed -- Visual Studio Build Tools, added to unblock several
other stubs) gets past the original `torch-scatter` build-isolation
failure entirely: `torch-scatter`/`torch-sparse`/`torch-cluster`/
`torch-spline-conv` all now prepare their metadata successfully (the
predicted "compiler-toolchain wall next" didn't materialize -- these
are legitimately buildable here now). Install fails for a completely
different reason instead: dependency resolution across every published
`fedgraph` version (0.0.9 through 0.2.5) shows they all require
`ray>=2.6.3`, and `ray` has no wheel at all for this Python version in
this environment -- the same root-cause blocker already documented on
distributed_frameworks/ray_train_adapter.py and
distributed_frameworks/alpa_adapter.py. Confirmed no side effects from
the attempt (torch/torchvision/numpy versions unchanged, nothing left
partially installed). Revisit once Ray ships a compatible wheel -- unlike
the original prediction, there's no compiler-toolchain wall left in the
way once that happens.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedGraph's transitive torch-scatter dependency fails to see the installed "
        "torch during its isolated build -- see this module's docstring."
    )


FL_FRAMEWORKS.register("FedGraph", implemented=False, organization="Rutgers University")(build)
