"""ResNet50 -- real implementation, same pattern as ResNet18: random init
(no pretrained download needed), num_classes=10 to match CIFAR10.
torchvision is imported lazily inside build() so registering this
architecture doesn't require it installed.
"""
from core.registry import ARCHITECTURES


def build(framework_adapter, config):
    import torchvision

    return torchvision.models.resnet50(weights=None, num_classes=10)


ARCHITECTURES.register(
    "ResNet50", implemented=True, family="CNN", framework="PyTorch", input_shape=(3, 32, 32)
)(build)
