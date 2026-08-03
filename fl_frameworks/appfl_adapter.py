"""APPFL -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("APPFL is not yet implemented.")


FL_FRAMEWORKS.register("APPFL", implemented=False, organization='Argonne National Laboratory')(build)
