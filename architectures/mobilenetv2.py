"""MobileNetV2 -- real implementation, same pattern as ResNet18: random
init (no pretrained download needed), num_classes=10 to match CIFAR10.
torchvision is imported lazily inside build() so registering this
architecture doesn't require it installed.

`also_supports` only lists OpenVINO, not ResNet18's full converter list --
that's specifically what was cross-checked directly (README.md's
MobileNetV2/ResNet50 bullet), not assumed by analogy. Both this and
ResNet50 are plain `nn.Module`s built the same way as ResNet18, so the
rest of ResNet18's converter list would likely also work, but "likely"
isn't "verified" -- this stays narrow to what's actually been run, the
same discipline `architectures/vit.py`'s docstring applies in the other
direction (explicitly documenting OpenVINO does NOT work for ViT, despite
looking like it should by the same analogy).
"""
from core.registry import ARCHITECTURES


def build(framework_adapter, config):
    import torchvision

    return torchvision.models.mobilenet_v2(weights=None, num_classes=10)


ARCHITECTURES.register(
    "MobileNetV2",
    implemented=True,
    family="CNN",
    framework="PyTorch",
    also_supports=["OpenVINO"],
    input_shape=(3, 32, 32),
)(build)
