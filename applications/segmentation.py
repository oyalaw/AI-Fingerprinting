"""Segmentation application -- real implementation, paired with
architectures/yolov8_seg.py. preprocess is identical to
applications/object_detection.py's (640x640 resize, 0-1 scaled RGB, no
ImageNet normalization -- same YOLO convention).

postprocess splits the flat wire tensor back into the two shapes
architectures/yolov8_seg.py's wrapper flattened (`DETECTIONS_SHAPE`,
`PROTOTYPES_SHAPE`, `DETECTIONS_NUMEL` imported directly from there so
the split point can never drift out of sync between the two files), then
runs Ultralytics' own real decode: `non_max_suppression(..., nc=80)` --
the `nc` is what tells NMS that only 80 of the post-box columns are class
scores and the remaining 32 are mask coefficients to carry through
untouched, rather than misreading all 112 as classes -- followed by
`process_mask` to linearly combine the 32 prototype masks per detection
into actual per-object masks. Cross-checked directly against Ultralytics'
own high-level `YOLO.predict()` on the same real image (Ultralytics'
bundled bus.jpg test asset) before writing this file: same detection
count, matching classes/confidences, and mask areas in the same relative
proportions (the bus mask by far the largest in both runs, as expected) --
not a hand-rolled reimplementation that happens to run without erroring.
"""
import numpy as np
import torch

from applications.base import Application
from architectures.yolov8_seg import DETECTIONS_NUMEL, DETECTIONS_SHAPE, PROTOTYPES_SHAPE
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


class Segmentation(Application):
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
        from ultralytics.utils.ops import process_mask

        flat = output_tensor.flatten()
        detections = flat[:DETECTIONS_NUMEL].reshape(DETECTIONS_SHAPE)
        prototypes = flat[DETECTIONS_NUMEL:].reshape(PROTOTYPES_SHAPE)

        nms_out = non_max_suppression(detections, conf_thres=0.25, iou_thres=0.45, nc=80)[0]
        boxes = nms_out[:, :4]
        mask_coeffs = nms_out[:, 6:]
        masks = process_mask(
            prototypes[0], mask_coeffs, boxes, (_INPUT_SIZE, _INPUT_SIZE), upsample=True
        )

        results = []
        for row, mask in zip(nms_out.tolist(), masks):
            # Each row is [x1, y1, x2, y2, conf, cls_id, *32 mask coefficients]
            # -- only the first 6 columns are needed here, process_mask()
            # already consumed the mask coefficients above.
            x1, y1, x2, y2, conf, cls_id = row[:6]
            cls_id = int(cls_id)
            results.append(
                {
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": _COCO_CLASSES[cls_id] if cls_id < len(_COCO_CLASSES) else str(cls_id),
                    "mask_area_px": int((mask > 0.5).sum().item()),
                }
            )
        return results


APPLICATIONS.register("Segmentation", implemented=True)(Segmentation)
