"""OpenFL -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("OpenFL is not yet implemented.")


FL_FRAMEWORKS.register("OpenFL", implemented=False, organization='Intel')(build)
