"""PaddleDetection -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("PaddleDetection is not yet implemented.")


CV_FRAMEWORKS.register("PaddleDetection", implemented=False, models='Detection')(build)
