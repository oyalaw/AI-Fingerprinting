"""Alpa -- registered but not yet implemented.

Checked directly: `pip install alpa` fails dependency resolution --
every published Alpa version pins an exact/bounded `ray` version
(`ray==2.1.0`, `ray==1.13.0`, ...), and `ray` itself has no wheel at all
for this project's Python version in this environment (same root cause as
distributed_frameworks/ray_train_adapter.py). Separately, even past that:
Alpa is JAX-native, the same structural mismatch documented in
fl_frameworks/fedjax_adapter.py -- models are JAX functions/pytrees, not
this project's shared PyTorch ResNet18, so a real adapter would need an
entirely separate hand-written architecture.

Re-checked on Ubuntu, since distributed_frameworks/ray_train_adapter.py's
own Ray-wheel wall resolved (ray 2.57.0 now ships a real cp313 wheel).
Surprising result: `pip install alpa` now actually succeeds at dependency
resolution -- it resolves current `ray==2.57.0`, not the old exact pin
this docstring's original check found (pip's resolver evidently picked a
looser constraint this time; not re-verified against every historical
Alpa release). But `import alpa` fails with a new, different, genuine
bug: `RuntimeError: jaxlib version 0.11.1 is newer than and incompatible
with jax version 0.3.15` -- Alpa's own dependency resolution pulled an old
`jax==0.3.15` (built from source, no prebuilt wheel for it either) paired
with a `jaxlib` far newer than that old jax release's own internal
version check tolerates, a real mismatch in Alpa's own pins, not this
project's adapter code. Confirmed no lasting side effects (reverted
alpa/jax/jaxlib/ray and everything else it pulled in; restored `rich` to
the version flwr needs, which the attempt had also downgraded).

Moot either way: the structural JAX-native mismatch above is unaffected
by any of this and remains the real, permanent reason this stays a stub,
the same standing as fl_frameworks/fedjax_adapter.py.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Alpa is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("Alpa", implemented=False, organization="Stanford")(build)
