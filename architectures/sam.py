"""SAM (Segment Anything Model) -- a new architecture entry beyond the
original 15, paired with applications/segmentation.py alongside
architectures/yolov8_seg.py. Family=Transformer (not CNN): SAM's image
encoder is explicitly a Vision Transformer backbone (MAE-pretrained ViT),
the same basis for classifying architectures/vit.py, not a conventional
CNN like YOLOv8.

Built via `segment_anything.sam_model_registry["vit_b"](checkpoint=None)`
-- confirmed directly this is a real, supported call (the function's own
default is `checkpoint=None`), giving a random-init 93.7M-param model, the
smallest of SAM's three official variants and consistent with every other
architecture here: no pretrained weights, the point is realistic traffic.
Its ViT image encoder hard-requires 1024x1024 input (`model.image_encoder.img_size`).

SAM is normally driven through the high-level `SamPredictor` (point/box
click prompts, multiple candidate masks), which doesn't fit a stateless
one-tensor-in/one-tensor-out request. Instead _SAMWrapper drives SAM's
three real internal stages directly -- confirmed each stage's signature
and chained them end-to-end before writing this file:

  1. `image_encoder(x)` -> `(1, 256, 64, 64)` image embedding (after the
     same `(x - pixel_mean) / pixel_std` normalization `Sam.forward` uses).
  2. `prompt_encoder(points=(coords, labels), boxes=None, masks=None)` --
     a fixed single foreground point at the image center stands in for a
     real click, since this project's dataset samples don't carry click
     annotations. Confirmed shapes: sparse `(1, 2, 256)`, dense
     `(1, 256, 64, 64)`.
  3. `mask_decoder(...)` with `multimask_output=False` (one mask, not
     SAM's usual three candidates) -> `low_res_masks (1, 1, 256, 256)` +
     `iou_predictions (1, 1)`.

Flattens and concatenates both outputs at a fixed, documented split point
(`MASK_NUMEL`), the same technique architectures/yolov8_seg.py uses for
its own two-tensor output -- applications/segmentation.py's postprocess
reverses it, dispatching on total element count (SAM's 65,537 vs
YOLOv8-Seg's 1,793,600 are unambiguous) rather than on which architecture
was selected, since Application instances aren't constructed with
architecture context.

applications/segmentation.py's shared preprocess() is fixed at 640x640
(YOLOv8-Seg's requirement, established first) -- rather than thread
architecture-specific target sizes through the application layer,
_SAMWrapper upscales whatever it's given to its own required 1024x1024
via bilinear interpolation. The model is random-init regardless, so
upscaled-then-encoded input costs nothing real; this keeps one shared
preprocess() correct for every Segmentation-paired architecture.
"""
import torch

from core.registry import ARCHITECTURES

MASK_SHAPE = (1, 1, 256, 256)
MASK_NUMEL = 1 * 1 * 256 * 256


class _SAMWrapper(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        image_size = self.sam_model.image_encoder.img_size
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = torch.nn.functional.interpolate(
                x, size=(image_size, image_size), mode="bilinear", align_corners=False
            )

        x = (x - self.sam_model.pixel_mean) / self.sam_model.pixel_std
        image_embeddings = self.sam_model.image_encoder(x)

        center = image_size / 2.0
        point_coords = torch.full((batch_size, 1, 2), center, device=x.device)
        point_labels = torch.ones((batch_size, 1), device=x.device)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        image_pe = self.sam_model.prompt_encoder.get_dense_pe()

        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return torch.cat([low_res_masks.flatten(), iou_predictions.flatten()]).unsqueeze(0)


def build(framework_adapter, config):
    from segment_anything import sam_model_registry

    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    return _SAMWrapper(sam_model)


ARCHITECTURES.register(
    "SAM",
    implemented=True,
    family="Transformer",
    framework="PyTorch",
    application="Segmentation",
    input_shape=(3, 1024, 1024),
)(build)
