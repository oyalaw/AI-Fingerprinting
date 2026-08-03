"""UCI HAR dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class UciHar(Dataset):
    def samples(self, n):
        raise NotImplementedError("UCI HAR is not yet implemented.")


DATASETS.register("UCI HAR", implemented=False)(UciHar)
