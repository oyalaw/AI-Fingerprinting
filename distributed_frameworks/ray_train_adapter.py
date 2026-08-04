"""Ray Train -- registered but not yet implemented.

Checked directly: `pip install ray` (and `ray[train]`) both fail with
`No matching distribution found` -- Ray has no published wheel at all for
this project's Python version in this environment. Nothing to build
against; revisit once Ray ships a compatible wheel (`pip index versions
ray` is the quick way to check).
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Ray Train is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("Ray Train", implemented=False, organization="Anyscale")(build)
