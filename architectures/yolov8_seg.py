"""YOLOv8-Seg -- a new architecture entry, not one of the original 12: the
taxonomy's single "YOLOv8" entry is genuinely a different checkpoint/head
from Ultralytics' segmentation variant (`yolov8n-seg.pt`, auto-downloaded,
~6.7MB), with a structurally different raw output -- same well-justified
gap-fill precedent as architectures/autoencoder.py. Paired with
applications/segmentation.py.

Confirmed directly (not assumed): eval-mode forward() on a 640x640 input
returns `((detections, mask_prototypes), features)`, where `detections` is
`(1, 116, 8400)` -- 4 box coords + 80 COCO class scores + 32 mask
coefficients per anchor, vs plain YOLOv8's `(1, 84, 8400)` -- and
`mask_prototypes` is `(1, 32, 160, 160)`, 32 prototype masks at 1/4 input
resolution that get linearly combined per-detection using its 32 mask
coefficients (Ultralytics' `process_mask`, invoked in
applications/segmentation.py's postprocess).

This project's wire protocol is one-tensor-in/one-tensor-out, but here
there are two structurally different output tensors to send. Rather than
introduce a second wire convention alongside applications/sentiment_analysis.py's
stacking approach (which only works because BERT's three inputs share one
shape), _YOLOv8SegWrapper flattens both tensors to 1D and concatenates
them at a fixed, documented split point -- applications/segmentation.py's
postprocess reverses this with the same two shapes, which are fixed for
any 640x640 input (this architecture's declared input_shape).
"""
import torch

from core.registry import ARCHITECTURES

DETECTIONS_SHAPE = (1, 116, 8400)
PROTOTYPES_SHAPE = (1, 32, 160, 160)
DETECTIONS_NUMEL = 1 * 116 * 8400


class _YOLOv8SegWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, x):
        output = self.yolo_model(x)
        (detections, mask_prototypes), _features = output
        return torch.cat([detections.flatten(), mask_prototypes.flatten()]).unsqueeze(0)


def build(framework_adapter, config):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n-seg.pt")
    return _YOLOv8SegWrapper(yolo.model)


ARCHITECTURES.register(
    "YOLOv8-Seg", implemented=True, family="CNN", framework="PyTorch", input_shape=(3, 640, 640)
)(build)
