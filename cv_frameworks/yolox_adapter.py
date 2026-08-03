"""YOLOX -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("YOLOX is not yet implemented.")


CV_FRAMEWORKS.register("YOLOX", implemented=False, models='Detection')(build)
