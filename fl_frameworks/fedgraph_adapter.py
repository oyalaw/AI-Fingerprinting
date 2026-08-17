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

That revisit condition is now met. Re-checked on Ubuntu: `ray` 2.57.0
ships a real `cp313` `manylinux2014_x86_64` wheel now (confirmed via
Ray's own PyPI file index -- this looks like Ray shipping a newer Python
version's wheel since the original check, not something OS-specific).
With that gone, `pip install --no-build-isolation fedgraph` genuinely
succeeds end-to-end this time: `torch-cluster`/`torch-scatter`/
`torch-sparse`/`torch-spline-conv` all actually compile for real (several
minutes of real `g++`/`cc1plus` C++ extension builds, confirmed directly
watching the process), and both `import fedgraph` and `import ray` work
cleanly afterward. Every wall this docstring traced -- build-isolation
blindness, the compiler-toolchain prediction, and Ray's wheel gap -- is
now resolved.

Reverted afterward rather than left installed: `fedgraph`'s own dependency
tree pulls in a large, unrelated surface for a graph-FL package (`ray`,
`sphinx`/`sphinx_rtd_theme`/`twine` -- its own docs/publishing tooling,
`tenseal`, `ogb`, `gdown`, ~50 packages total) and downgrades `pyyaml` to
6.0.1 in a way that conflicts with `flwr`'s own `pyyaml<7.0.0,>=6.0.2`
requirement -- confirmed directly this breaks pip's own dependency
resolution once both are installed together. Since this project's own
`config.yaml` parsing depends on `pyyaml` project-wide, leaving that
conflict in place would be exactly the kind of unprompted side effect this
project's setuptools-pin investigations were careful to avoid (see
cv_frameworks/mmdetection_adapter.py's docstring) -- uninstalled fedgraph
and its unique dependency tree, restored `pyyaml==6.0.3`, confirmed
`python main.py --list` and the rest of this project's already-verified
stack still work unchanged. What's left to actually implement this
adapter is real engineering (FedGraph's own federated GNN training API,
distinct from this project's PyTorch/CIFAR10 classification loop every
other FL adapter shares), not an environment blocker -- left as a stub for
that reason.

Investigated exactly how much engineering, rather than leaving that as a
vague estimate: `fedgraph`'s real entry points are task-specific --
`fedgraph.federated_methods.run_NC`/`run_GC`/`run_LP` (node
classification / graph classification / link prediction), each with its
own `Trainer_*`/`Server_*` classes. Confirmed directly, none of them take
a dataset object -- `fedgraph.data_process.data_loader(args:
attridict.AttriDict)` expects an `args` config naming one of FedGraph's
own bundled benchmark datasets (citation graphs like Cora/CiteSeer for
`run_NC`, TU datasets like MUTAG/PROTEINS for `run_GC`, MovieLens for
`run_LP`) and returns FedGraph's own graph-partitioned data structures,
not something this project's `datasets/karate_club.py` (or any other
registry entry) can be swapped in for. `federated_methods` also imports
`ray` directly and builds its trainers as Ray actors internally -- its
own distribution mechanism, unrelated to
distributed_frameworks/ray_train_adapter.py's `torch.distributed`-based
one, or to any FL adapter here.

This is the same structural wall README.md's own FL/distributed
generalization section already documents for Node Classification/GCN in
this project's *own* code ("this needs a fundamentally different
single-graph training loop, not a generalized version of the per-sample
one") -- FedGraph's `run_NC` would hit exactly that wall, plus wants its
own benchmark dataset story on top. Concretely larger than
distributed_frameworks/ray_train_adapter.py's or
frameworks/tvm_adapter.py's rewrites (each a few hours against a stable,
single-purpose API): a real FedGraph adapter means either building a new
graph-dataset/transductive-training pipeline this project doesn't have at
all, or adopting FedGraph's own benchmark datasets and accepting they
won't fit this project's existing 5-level ground-truth taxonomy the way
every other dataset here does. Left as a stub for that reason, tracked
here with the concrete scope rather than a vague "needs more work."
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedGraph's install wall (Ray's missing wheel) is resolved on Ubuntu -- what's "
        "left is writing a real adapter for its own federated-GNN API, not an "
        "environment blocker -- see this module's docstring."
    )


FL_FRAMEWORKS.register("FedGraph", implemented=False, organization="Rutgers University")(build)
