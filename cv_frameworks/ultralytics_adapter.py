"""Ultralytics -- registered but not yet implemented."""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Ultralytics is not yet implemented.")


CV_FRAMEWORKS.register("Ultralytics", implemented=False, models='YOLOv5, YOLOv8, YOLOv11')(build)
