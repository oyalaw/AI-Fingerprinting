"""UCI HAR (Human Activity Recognition Using Smartphones) -- the real UCI
dataset, downloaded and parsed directly with the standard library
(urllib/zipfile). Verified directly against the real archive before
writing this: it's a nested zip (the outer .zip contains a second
`UCI HAR Dataset.zip`), with 9 per-channel "Inertial Signals" text files
(body_acc/body_gyro/total_acc, each x/y/z), one row per sample window,
128 whitespace-separated scientific-notation floats per row (a 128-
timestep window), confirmed 2947 samples in the test split.
Activity labels (1-6 in y_{split}.txt) mapped to names via
activity_labels.txt: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING.

Source: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip,
~60MB, UCI's own current canonical URL for this dataset. Downloaded once
to ./data/UCI_HAR and cached, same pattern as CIFAR10/IMDB's own on-disk
caches.
"""
import pathlib
import urllib.request
import zipfile

from core.registry import DATASETS
from datasets.base import Dataset

_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

_CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)

_ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


class UCIHARDataset(Dataset):
    def __init__(self, root="./data", split="test"):
        self.root = pathlib.Path(root)
        self.split = split
        self._extracted = self.root / "UCI_HAR" / "UCI HAR Dataset"
        self._ensure_downloaded()

        split_dir = self._extracted / split
        signals_dir = split_dir / "Inertial Signals"
        self._channel_lines = [
            (signals_dir / f"{channel}_{split}.txt").read_text().splitlines()
            for channel in _CHANNELS
        ]
        self._labels = [int(v) for v in (split_dir / f"y_{split}.txt").read_text().split()]

    def _ensure_downloaded(self):
        if self._extracted.exists():
            return
        target_dir = self.root / "UCI_HAR"
        target_dir.mkdir(parents=True, exist_ok=True)

        archive_path = target_dir / "uci_har.zip"
        if not archive_path.exists():
            urllib.request.urlretrieve(_ARCHIVE_URL, archive_path)

        with zipfile.ZipFile(archive_path) as outer:
            inner_bytes = outer.read("UCI HAR Dataset.zip")
        inner_path = target_dir / "UCI HAR Dataset.zip"
        inner_path.write_bytes(inner_bytes)
        with zipfile.ZipFile(inner_path) as inner:
            inner.extractall(target_dir)

    def samples(self, n):
        import numpy as np

        count = min(n, len(self._labels))
        for i in range(count):
            window = np.array(
                [[float(v) for v in self._channel_lines[c][i].split()] for c in range(len(_CHANNELS))],
                dtype=np.float32,
            )  # (9, 128) -- (channels, timesteps)
            yield window, _ACTIVITY_LABELS[self._labels[i]]


DATASETS.register("UCI HAR", implemented=True)(UCIHARDataset)
