"""NVFlare -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("NVFlare is not yet implemented.")


FL_FRAMEWORKS.register("NVFlare", implemented=False, organization='NVIDIA')(build)
