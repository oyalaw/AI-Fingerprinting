"""CNN family: shared metadata + image preprocessing reused by every CNN
architecture (ResNet18, ResNet50, MobileNetV2, YOLOv8, ...). torch is
imported lazily so registering this family doesn't require it installed."""
from core.registry import FAMILIES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_chw(array_hwc_uint8):
    """(H, W, 3) uint8 -> normalized (3, H, W) float32 torch tensor."""
    import torch

    tensor = torch.from_numpy(array_hwc_uint8.copy()).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


FAMILIES.add("CNN", implemented=True, description="Convolutional Neural Networks")
