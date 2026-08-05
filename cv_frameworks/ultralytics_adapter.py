"""Ultralytics -- real, but still informational even after config is
threaded through to every architecture's build() (see architectures/gcn.py
and architectures/ddpm.py for the two entries where that threading enables
a genuine dispatch): there's nothing for YOLOv8 to dispatch *to* here.
architectures/yolov8.py loads YOLOv8 via Ultralytics itself
(`ultralytics.YOLO("yolov8n.pt")`) unconditionally -- Ultralytics *is* the
reference implementation of YOLOv8, so selecting `framework=PyTorch,
architecture=YOLOv8` already means "run it via Ultralytics," whether or
not `cv_framework=Ultralytics` is separately set. This entry documents
that fact and is registered `implemented` for that reason, not because
setting it changes anything at runtime -- there's no alternative CV
framework implemented yet for it to meaningfully switch to.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    return "ultralytics"


CV_FRAMEWORKS.register(
    "Ultralytics", implemented=True, models="YOLOv5, YOLOv8, YOLOv11"
)(build)
