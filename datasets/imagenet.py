"""ImageNet dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class Imagenet(Dataset):
    def samples(self, n):
        raise NotImplementedError("ImageNet is not yet implemented.")


DATASETS.register("ImageNet", implemented=False)(Imagenet)
