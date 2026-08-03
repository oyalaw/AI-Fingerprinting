"""Clara FL -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Clara FL is not yet implemented.")


FL_FRAMEWORKS.register("Clara FL", implemented=False, organization='NVIDIA')(build)
