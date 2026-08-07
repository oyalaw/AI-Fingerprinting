"""Object Detection application -- paired with two architectures now,
architectures/yolov8.py and architectures/detectron2.py, with genuinely
different raw output formats (YOLOv8: dense (1, 84, 8400) anchor-grid
tensor needing NMS applied here; Detectron2: already-decoded, fixed-shape
(1, 100, 6) padded detections -- its own RPN + ROI heads + per-class NMS
already ran inside the model). Application instances are built with no
architecture context, so postprocess() dispatches on total element count,
the same disambiguation applications/segmentation.py already uses between
SAM and YOLOv8-Seg: YOLOv8's is always 705,600 (84*8400), Detectron2's is
always 600 (100*6) -- imported directly from architectures/detectron2.py
so the two files can't drift out of sync.

preprocess resizes/normalizes a raw HWC uint8 image to a 640x640 CHW float
input (no ImageNet mean/std normalization -- both architectures just want
0-1 scaled RGB, the YOLO convention, unlike the CNN classification
convention in families/cnn/__init__.py) -- shared by both architectures.

YOLOv8's branch: runs Ultralytics' own `non_max_suppression` on the raw
detection tensor -- verified directly against Ultralytics' own high-level
`YOLO.predict()` pipeline on the same image: same box region/class, same
decode logic, not a hand-rolled reimplementation.

Detectron2's branch: just reads the already-decoded
[x1, y1, x2, y2, score, class_id] rows back out, skipping padded
(all-zero) rows -- see architectures/detectron2.py's docstring for how
Detectron2's variable-count Instances output gets packed into this fixed
shape in the first place.
"""
import numpy as np
import torch

from applications.base import Application
from architectures.detectron2 import DETECTIONS_NUMEL as D2_DETECTIONS_NUMEL, DETECTIONS_SHAPE as D2_DETECTIONS_SHAPE
from core.registry import APPLICATIONS

_INPUT_SIZE = 640
_YOLOV8_TOTAL_NUMEL = 84 * 8400

_COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class ObjectDetection(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            array = raw_sample
            if array.shape[:2] != (_INPUT_SIZE, _INPUT_SIZE):
                from PIL import Image

                array = np.array(Image.fromarray(array).resize((_INPUT_SIZE, _INPUT_SIZE)))
            tensor = torch.from_numpy(array.copy()).float().permute(2, 0, 1) / 255.0
            return tensor
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        numel = output_tensor.numel()
        if numel == D2_DETECTIONS_NUMEL:
            return self._postprocess_detectron2(output_tensor)
        if numel == _YOLOV8_TOTAL_NUMEL:
            return self._postprocess_yolov8(output_tensor)
        raise ValueError(
            f"ObjectDetection.postprocess got {numel} elements, matching neither "
            f"YOLOv8 ({_YOLOV8_TOTAL_NUMEL}) nor Detectron2 ({D2_DETECTIONS_NUMEL})."
        )

    def _postprocess_yolov8(self, output_tensor):
        from ultralytics.utils.nms import non_max_suppression

        detections = non_max_suppression(output_tensor, conf_thres=0.25, iou_thres=0.45)[0]
        results = []
        for x1, y1, x2, y2, conf, cls_id in detections.tolist():
            cls_id = int(cls_id)
            results.append(
                {
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": _COCO_CLASSES[cls_id] if cls_id < len(_COCO_CLASSES) else str(cls_id),
                }
            )
        return results

    def _postprocess_detectron2(self, output_tensor):
        rows = output_tensor.reshape(D2_DETECTIONS_SHAPE)
        results = []
        for x1, y1, x2, y2, conf, cls_id in rows.tolist():
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0 and conf == 0:
                continue  # padded row, not a real detection
            cls_id = int(cls_id)
            results.append(
                {
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": _COCO_CLASSES[cls_id] if cls_id < len(_COCO_CLASSES) else str(cls_id),
                }
            )
        return results


APPLICATIONS.register(
    "Object Detection", implemented=True, datasets=["Synthetic", "COCO"]
)(ObjectDetection)
