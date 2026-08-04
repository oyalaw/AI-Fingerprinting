"""Object Detection application -- real implementation, paired with
architectures/yolov8.py. preprocess resizes/normalizes a raw HWC uint8
image to YOLOv8's expected 640x640 CHW float input (no ImageNet
mean/std normalization -- YOLO just wants 0-1 scaled RGB, unlike the CNN
classification convention in families/cnn/__init__.py). postprocess runs
Ultralytics' own `non_max_suppression` on the raw (1, 84, 8400) detection
tensor -- verified directly against Ultralytics' own high-level
`YOLO.predict()` pipeline on the same image: same box region/class,
same decode logic, not a hand-rolled reimplementation.
"""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_INPUT_SIZE = 640

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


APPLICATIONS.register("Object Detection", implemented=True)(ObjectDetection)
