"""LEAF -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("LEAF is not yet implemented.")


FL_FRAMEWORKS.register("LEAF", implemented=False, organization='Carnegie Mellon University')(build)
