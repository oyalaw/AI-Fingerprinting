"""Segment Anything -- real, but still informational, same nuance as
cv_frameworks/ultralytics_adapter.py: architectures/sam.py builds the
model via `segment_anything.sam_model_registry` unconditionally --
`segment-anything` *is* the reference implementation of SAM, so selecting
`framework=PyTorch, architecture=SAM` already means "run it via Meta's
segment-anything package," whether or not `cv_framework=Segment Anything`
is separately set. This entry documents that fact and is registered
`implemented` for that reason, not because setting it changes anything at
runtime.
"""
from core.registry import CV_FRAMEWORKS


def build(**kwargs):
    return "segment_anything"


CV_FRAMEWORKS.register(
    "Segment Anything", implemented=True, models="Segmentation"
)(build)
