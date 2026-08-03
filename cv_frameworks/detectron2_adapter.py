"""Detectron2 -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Detectron2 is not yet implemented.")


CV_FRAMEWORKS.register("Detectron2", implemented=False, models='Meta vision models')(build)
