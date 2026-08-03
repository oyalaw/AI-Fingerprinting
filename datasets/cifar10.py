"""CIFAR10 -- the one fully-implemented dataset this pass. Downloads via
torchvision on first use (or reuses an existing local copy under ./data).
torchvision is imported lazily inside __init__ so registering this dataset
doesn't require it installed."""
import numpy as np

from core.registry import DATASETS
from datasets.base import Dataset

_CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class CIFAR10Dataset(Dataset):
    def __init__(self, root="./data", download=True):
        import torchvision

        self._torch_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=download)

    def samples(self, n):
        count = min(n, len(self._torch_dataset))
        for i in range(count):
            pil_image, label_index = self._torch_dataset[i]
            array = np.array(pil_image)  # (32, 32, 3) uint8
            yield array, _CIFAR10_CLASSES[label_index]


DATASETS.register("CIFAR10", implemented=True)(CIFAR10Dataset)
