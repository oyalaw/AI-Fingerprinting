"""SST-2 (Stanford Sentiment Treebank v2, GLUE benchmark release) --
downloaded and parsed directly with the standard library (urllib/zipfile/
csv), same reasoning as datasets/imdb.py: HuggingFace's `datasets` pip
package's own top-level import name collides with this project's own
datasets/ package.

Source: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip (Facebook's
GLUE data mirror), ~7.4MB. Confirmed directly by downloading and
inspecting it: dev.tsv/train.tsv are `sentence\\tlabel\\n` (label 0/1);
test.tsv has no labels at all -- it's the GLUE leaderboard's held-out
submission file, unusable here. dev.tsv is used instead, the same
"held-out labeled split" role CIFAR10's train=False and IMDB's test/
directory play.
"""
import csv
import pathlib
import urllib.request
import zipfile

from core.registry import DATASETS
from datasets.base import Dataset

_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
_LABELS = ("negative", "positive")


class SST2Dataset(Dataset):
    def __init__(self, root="./data"):
        self.root = pathlib.Path(root)
        self._ensure_downloaded()

    def _ensure_downloaded(self):
        extracted = self.root / "SST-2"
        if extracted.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / "SST-2.zip"
        if not archive_path.exists():
            urllib.request.urlretrieve(_ARCHIVE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(self.root)

    def samples(self, n):
        dev_path = self.root / "SST-2" / "dev.tsv"
        with dev_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= n:
                    break
                yield row["sentence"].strip(), _LABELS[int(row["label"])]


DATASETS.register("SST2", implemented=True)(SST2Dataset)
