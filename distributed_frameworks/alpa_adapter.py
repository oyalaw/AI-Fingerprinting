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
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Alpa is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("Alpa", implemented=False, organization="Stanford")(build)
