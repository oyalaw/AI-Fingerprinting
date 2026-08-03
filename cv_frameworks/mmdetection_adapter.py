"""MMDetection -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("MMDetection is not yet implemented.")


CV_FRAMEWORKS.register("MMDetection", implemented=False, models='Detection models')(build)
