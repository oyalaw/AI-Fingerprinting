"""Detectron2 -- a new architecture entry, not a cv_framework dispatch
under the existing YOLOv8 architecture, following the same reasoning
cv_frameworks/yolox_adapter.py's docstring already lays out for YOLOX:
Detectron2's Faster R-CNN (two-stage: region proposals -> per-region
classification/box regression) is a genuinely different detector design
from YOLOv8 (one-stage, dense anchor grid), not an alternate
implementation of the same architecture the way
graph_frameworks/pytorch_geometric_adapter.py is an alternate GCN.

Previously a stub: `pip install detectron2` returns no matching
distribution (never published to PyPI). Retried after installing a real
C/C++ compiler (Visual Studio Build Tools, installed to unblock several
other stubs sharing a similar root cause):
`pip install git+https://github.com/facebookresearch/detectron2.git`
(Detectron2's own documented source-install path) now succeeds --
confirmed directly: it compiles Detectron2's custom ops as a real CPU-only
extension (no CUDA toolkit/nvcc on this machine, and Detectron2's own
setup.py falls back to a CPU-only build in that case rather than failing).

`detectron2.model_zoo` itself is NOT usable here -- confirmed directly it
hits the same `ModuleNotFoundError: No module named 'pkg_resources'` this
project already found for `cv_frameworks/mmdetection_adapter.py` and
`fl_frameworks/pysyft_adapter.py` (pkg_resources removed from this
environment's modern setuptools). Routed around it the same way this
project avoids pretrained-weight downloads generally: `detectron2.config
.get_cfg()` + `detectron2.modeling.build_model(cfg)` build a real,
random-init `GeneralizedRCNN` directly, no model_zoo config file or
checkpoint needed. Confirmed directly: default config's ResNet
backbone needs `MODEL.RESNETS.RES2_OUT_CHANNELS = 64` explicitly set for
the R18/R34 depths (`build_resnet_backbone` asserts this) -- not set
automatically from `RESNETS.DEPTH` alone, a real Detectron2 config
gotcha, not a bug in this project.

Confirmed directly the real forward pass shape/format: eval-mode
`GeneralizedRCNN.__call__` takes a list of `{"image": <uint8 CHW tensor>,
"height": int, "width": int}` dicts (not a plain batched tensor the way
every other architecture in this project works), and returns a list of
`{"instances": Instances}` with real, already-decoded `pred_boxes` /
`scores` / `pred_classes` fields -- Detectron2's own RPN + ROI heads +
per-class NMS already ran internally, unlike YOLOv8's raw anchor-grid
output which needs `non_max_suppression` applied afterward in
`applications/object_detection.py`. `_Detectron2Wrapper` below absorbs
both of those differences: it builds the list-of-dicts input from a plain
`(1, 3, H, W)` float tensor internally (same input convention as
`architectures/yolov8.py`), and packs Detectron2's variable-count
`Instances` output into this project's fixed-shape one-tensor wire
contract by padding/truncating to a fixed `_MAX_DETECTIONS` rows of
`[x1, y1, x2, y2, score, class_id]`, zero-padded past the real detection
count (`ROI_HEADS.SCORE_THRESH_TEST = 0.0` confirmed directly this can
produce up to `ROI_HEADS.DETECTIONS_PER_IMAGE` = 100 real detections on a
random-init model, so padding is the common case, not an edge case).
`applications/object_detection.py`'s `postprocess` dispatches between this
fixed-count format and YOLOv8's dense-grid format by total element count,
the same disambiguation `applications/segmentation.py` already uses
between SAM and YOLOv8-Seg.
"""
import torch

from core.registry import ARCHITECTURES

_MAX_DETECTIONS = 100
_FIELDS_PER_DETECTION = 6  # x1, y1, x2, y2, score, class_id
DETECTIONS_SHAPE = (_MAX_DETECTIONS, _FIELDS_PER_DETECTION)
DETECTIONS_NUMEL = _MAX_DETECTIONS * _FIELDS_PER_DETECTION


class _Detectron2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size, _, height, width = x.shape

        inputs = [
            {"image": (x[i] * 255).to(torch.uint8), "height": height, "width": width}
            for i in range(batch_size)
        ]
        outputs = self.model(inputs)

        packed = torch.zeros(batch_size, *DETECTIONS_SHAPE, dtype=torch.float32)
        for i, out in enumerate(outputs):
            instances = out["instances"]
            n = min(len(instances), _MAX_DETECTIONS)
            if n == 0:
                continue
            packed[i, :n, 0:4] = instances.pred_boxes.tensor[:n]
            packed[i, :n, 4] = instances.scores[:n]
            packed[i, :n, 5] = instances.pred_classes[:n].float()
        return packed


def build(framework_adapter, config):
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = _MAX_DETECTIONS

    model = build_model(cfg)
    return _Detectron2Wrapper(model)


ARCHITECTURES.register(
    "Detectron2",
    implemented=True,
    family="CNN",
    framework="PyTorch",
    application="Object Detection",
    input_shape=(3, 640, 640),
)(build)
