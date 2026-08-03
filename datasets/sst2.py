"""SST2 dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class Sst2(Dataset):
    def samples(self, n):
        raise NotImplementedError("SST2 is not yet implemented.")


DATASETS.register("SST2", implemented=False)(Sst2)
