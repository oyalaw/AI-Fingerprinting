"""YOLOv8 -- real implementation, loaded from Ultralytics' own pretrained
weights (`yolov8n.pt`, auto-downloaded on first use, ~6MB). Ultralytics
*is* the reference implementation of YOLOv8, so "load YOLOv8" and "use
Ultralytics" are the same action here -- see cv_frameworks/ultralytics_adapter.py
for the corresponding cv_framework registration and applications/object_detection.py
for the raw-output decode step this feeds into.

Returns a plain nn.Module so frameworks/pytorch_adapter.py's existing
serialize/predict/deserialize handles it completely unchanged, same as
ResNet18 -- no changes anywhere in the generic framework machinery.
The one wrinkle, handled locally here rather than in the shared adapter:
YOLOv8's raw forward() returns a `(detections, features)` tuple in eval
mode (confirmed directly -- `detections` shape `(1, 84, 8400)` for a
640x640 input: 4 box coords + 80 COCO classes, 8400 anchor points across
3 detection scales), not a plain tensor, which would break
PyTorchAdapter.predict()'s `.cpu()` call. `_YOLOv8Wrapper` below unwraps
that tuple so the rest of the pipeline never needs to know about it.
"""
import torch

from core.registry import ARCHITECTURES


class _YOLOv8Wrapper(torch.nn.Module):
    """Unwraps YOLOv8's (detections, features) eval-mode output to just
    the detections tensor, so it's indistinguishable from any other
    single-tensor-output model to the rest of this project's pipeline."""

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, x):
        output = self.yolo_model(x)
        return output[0] if isinstance(output, (tuple, list)) else output


def build(framework_adapter):
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    return _YOLOv8Wrapper(yolo.model)


ARCHITECTURES.register(
    "YOLOv8", implemented=True, family="CNN", framework="PyTorch", input_shape=(3, 640, 640)
)(build)
