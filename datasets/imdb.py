"""IMDB dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class Imdb(Dataset):
    def samples(self, n):
        raise NotImplementedError("IMDB is not yet implemented.")


DATASETS.register("IMDB", implemented=False)(Imdb)
