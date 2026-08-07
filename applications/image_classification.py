"""Image classification application: raw HWC uint8 image -> normalized CHW
tensor in, predicted class index out. torch is imported lazily so
registering this application doesn't require it installed."""
import numpy as np

from applications.base import Application
from core.registry import APPLICATIONS
from families.cnn import normalize_chw


class ImageClassification(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return normalize_chw(raw_sample)
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        import torch

        probs = torch.softmax(output_tensor, dim=-1)
        return int(torch.argmax(probs, dim=-1).item())


APPLICATIONS.register(
    "Image Classification", implemented=True, datasets=["CIFAR10", "Synthetic", "ImageNet"]
)(ImageClassification)
