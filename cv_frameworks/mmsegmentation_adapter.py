"""MMSegmentation -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("MMSegmentation is not yet implemented.")


CV_FRAMEWORKS.register("MMSegmentation", implemented=False, models='Segmentation')(build)
