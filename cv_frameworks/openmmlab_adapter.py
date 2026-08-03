"""OpenMMLab -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("OpenMMLab is not yet implemented.")


CV_FRAMEWORKS.register("OpenMMLab", implemented=False, models='Vision toolkit')(build)
