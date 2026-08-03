"""Synthetic dataset: random tensors, no download required. Framework/
architecture-agnostic -- useful for smoke-testing any new combo before
wiring up a real dataset for it."""
import numpy as np

from core.registry import DATASETS
from datasets.base import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, shape=(32, 32, 3), seed=0):
        self.shape = shape
        self._rng = np.random.default_rng(seed)

    def samples(self, n):
        for i in range(n):
            array = self._rng.integers(0, 256, size=self.shape, dtype=np.uint8)
            yield array, f"synthetic-{i % 10}"


DATASETS.register("Synthetic", implemented=True)(SyntheticDataset)
