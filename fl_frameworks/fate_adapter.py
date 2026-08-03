"""FATE -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FATE is not yet implemented.")


FL_FRAMEWORKS.register("FATE", implemented=False, organization='WeBank')(build)
