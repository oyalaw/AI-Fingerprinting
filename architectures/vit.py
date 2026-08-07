"""ViT (Vision Transformer) -- reuses the existing Image Classification/
CIFAR10 pipeline, same as every CNN architecture; the first Transformer-
family architecture. torchvision's vit_b_16 hardcodes image_size=224 by
default (its positional embeddings are sized for a 224x224 input, and
break on anything else), but accepts an explicit image_size= override at
construction -- confirmed directly: vit_b_16(weights=None, num_classes=10,
image_size=32) accepts a (1, 3, 32, 32) input and produces the correct
(1, 10) output. Random init (no pretrained download), same reasoning as
every other architecture here: the point is the traffic, not accuracy.

Verified working end-to-end through frameworks/pytorch_adapter.py (its
native framework="PyTorch"). Deliberately NOT claiming OpenVINO support
the way architectures/resnet18.py does for its converter frameworks: tried
it directly and frameworks/openvino_adapter.py's conversion fails here --
`ov.convert_model`'s internal `torch.jit.trace(..., strict=False)` retraces
the model twice to check determinism, and ViT's Dropout layers get
different internal temporary-variable IDs on the two passes (visible in
the diff as e.g. `%78`/`%79` vs `%71`/`%72` -- shifted IDs, not a real
numerical difference), which trips OpenVINO's trace-consistency check as a
false positive. Not investigated further; OpenVINO's own error message
suggests retrying without `example_input` (scripted mode) as a workaround,
untested here since openvino_adapter.py's conversion path is shared by
every other architecture and changing it isn't safe to do for one
architecture's sake without re-verifying the rest.
"""
from core.registry import ARCHITECTURES


def build(framework_adapter, config):
    import torchvision

    return torchvision.models.vit_b_16(weights=None, num_classes=10, image_size=32)


ARCHITECTURES.register(
    "ViT",
    implemented=True,
    family="Transformer",
    framework="PyTorch",
    application="Image Classification",
    input_shape=(3, 32, 32),
)(build)
