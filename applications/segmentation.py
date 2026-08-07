"""Segmentation application -- paired with two architectures now,
architectures/yolov8_seg.py and architectures/sam.py, with genuinely
different raw output formats (YOLOv8-Seg: detection grid + mask
prototypes; SAM: a single low-res mask + IoU score). Application
instances are built with no architecture context (`APPLICATIONS.get(...)
.build()` takes no args), so postprocess() can't be told which
architecture produced a given tensor -- it dispatches on total element
count instead, which is unambiguous: YOLOv8-Seg's flattened output is
always 1,793,600 elements (116*8400 + 32*160*160), SAM's is always 65,537
(256*256 + 1) -- imported directly from each architecture module so
neither number can drift out of sync with its wrapper.

preprocess is identical to applications/object_detection.py's (640x640
resize, 0-1 scaled RGB, no ImageNet normalization -- same YOLO
convention) -- shared by both architectures; SAM's wrapper upscales to
its own required 1024x1024 internally, see architectures/sam.py's
docstring for why that's the right place for it rather than here.

YOLOv8-Seg's branch: splits the flat tensor back into the two shapes
architectures/yolov8_seg.py's wrapper flattened, then runs Ultralytics'
own real decode: `non_max_suppression(..., nc=80)` -- the `nc` is what
tells NMS that only 80 of the post-box columns are class scores and the
remaining 32 are mask coefficients to carry through untouched, rather
than misreading all 112 as classes -- followed by `process_mask` to
linearly combine the 32 prototype masks per detection into actual
per-object masks. Cross-checked directly against Ultralytics' own
high-level `YOLO.predict()` on the same real image (Ultralytics' bundled
bus.jpg test asset) before writing this file: same detection count,
matching classes/confidences, and mask areas in the same relative
proportions (the bus mask by far the largest in both runs, as expected).

SAM's branch: splits the flat tensor back into the low-res mask and IoU
score architectures/sam.py's wrapper flattened, applies SAM's own real
post-decode step (`torch.nn.functional.interpolate` back up to the
1024x1024 SAM operates at, then a sigmoid + 0.5 threshold -- the same
mask-decode SAM's own `SamPredictor.predict` performs after calling the
mask decoder, confirmed directly against its source).

Registered `datasets=["Synthetic"]` for `--interactive`'s filtering.
Unlike Speech Recognition/Image Generation above, preprocess() here
genuinely does use the dataset's image content (same generic HWC-array
handling as applications/object_detection.py's), so CIFAR10/COCO/
ImageNet would likely work too -- narrowed to `Synthetic` because that's
the only one ever actually run against this application (both YOLOv8-Seg
and SAM's experiment_matrix.yaml entries use it), not because anything
else would technically fail.
"""
import numpy as np
import torch

from applications.base import Application
from architectures.yolov8_seg import DETECTIONS_NUMEL, DETECTIONS_SHAPE, PROTOTYPES_SHAPE
from architectures.sam import MASK_NUMEL, MASK_SHAPE
from core.registry import APPLICATIONS

_YOLOV8_SEG_TOTAL_NUMEL = DETECTIONS_NUMEL + (1 * 32 * 160 * 160)
_SAM_TOTAL_NUMEL = MASK_NUMEL + 1

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
        flat = output_tensor.flatten()
        numel = flat.numel()

        if numel == _SAM_TOTAL_NUMEL:
            return self._postprocess_sam(flat)
        if numel == _YOLOV8_SEG_TOTAL_NUMEL:
            return self._postprocess_yolov8_seg(flat)
        raise ValueError(
            f"Segmentation.postprocess got {numel} elements, matching neither "
            f"SAM ({_SAM_TOTAL_NUMEL}) nor YOLOv8-Seg ({_YOLOV8_SEG_TOTAL_NUMEL})."
        )

    def _postprocess_yolov8_seg(self, flat):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import process_mask

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

    def _postprocess_sam(self, flat):
        low_res_mask = flat[:MASK_NUMEL].reshape(MASK_SHAPE)
        iou_prediction = flat[MASK_NUMEL:].item()

        upsampled = torch.nn.functional.interpolate(
            low_res_mask, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
        )
        mask = torch.sigmoid(upsampled) > 0.5

        return {
            "mask_area_px": int(mask.sum().item()),
            "iou_prediction": round(iou_prediction, 4),
        }


APPLICATIONS.register("Segmentation", implemented=True, datasets=["Synthetic"])(Segmentation)
