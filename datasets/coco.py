"""COCO dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class Coco(Dataset):
    def samples(self, n):
        raise NotImplementedError("COCO is not yet implemented.")


DATASETS.register("COCO", implemented=False)(Coco)
