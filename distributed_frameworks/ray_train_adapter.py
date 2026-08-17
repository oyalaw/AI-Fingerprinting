"""Ray Train -- registered but not yet implemented.

Checked directly: `pip install ray` (and `ray[train]`) both fail with
`No matching distribution found` -- Ray has no published wheel at all for
this project's Python version in this environment. Nothing to build
against; revisit once Ray ships a compatible wheel (`pip index versions
ray` is the quick way to check).

Re-checked on Ubuntu: resolved. Ray 2.57.0 ships a real `cp313`
`manylinux2014_x86_64` wheel -- confirmed directly, `pip install
ray[train]` and `import ray.train` both work cleanly (also pulls a
current `pyarrow` 25.0.1, itself now shipping a real `cp313` wheel, unlike
the old `pyarrow==17.0.0` pin fl_frameworks/pysyft_adapter.py is still
blocked on). Reverted afterward rather than left installed, since this
adapter isn't written yet -- what's left is a real adapter against Ray
Train's own `TorchTrainer`/`ScalingConfig` API (a different shape from
this project's other distributed adapters, which all layer directly on
`torch.distributed`), not an environment blocker anymore. See
fl_frameworks/fedgraph_adapter.py's docstring for the same Ray-wheel
finding from the FL side.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Ray's wheel wall is resolved on Ubuntu -- what's left is writing a real "
        "adapter against Ray Train's TorchTrainer API, not an environment blocker -- "
        "see this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("Ray Train", implemented=False, organization="Anyscale")(build)
