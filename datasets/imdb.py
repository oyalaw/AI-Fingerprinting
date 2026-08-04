"""IMDB -- the real Stanford Large Movie Review Dataset (Maas et al. 2011),
downloaded and parsed directly with the standard library (urllib/tarfile),
NOT via HuggingFace's `datasets` pip package.

That package's own top-level import name collides with this project's own
`datasets/` package -- this project's directory tree (approved at the very
start of this project) already uses `datasets/` for the registry, so
`import datasets` from anywhere inside this project resolves to this
package, not the pip-installed one. Confirmed directly: even after
`pip install datasets`, `datasets.__file__` evaluated from this project's
directory resolves to this package's own __init__.py, not site-packages;
from a neutral cwd it correctly resolves to the real library. There's no
clean fix short of renaming this project's own `datasets/` package, which
isn't worth doing for one dataset -- so IMDB is loaded via a small manual
downloader instead, using only the standard library.

Source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz,
~80MB, the standard/original release of this dataset. Downloaded once to
./data/aclImdb and cached, same pattern as CIFAR10's own on-disk cache.
"""
import pathlib
import tarfile
import urllib.request

from core.registry import DATASETS
from datasets.base import Dataset

_ARCHIVE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


class IMDBDataset(Dataset):
    def __init__(self, root="./data", split="test"):
        self.root = pathlib.Path(root)
        self.split = split
        self._ensure_downloaded()

    def _ensure_downloaded(self):
        extracted = self.root / "aclImdb"
        if extracted.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / "aclImdb_v1.tar.gz"
        if not archive_path.exists():
            urllib.request.urlretrieve(_ARCHIVE_URL, archive_path)
        with tarfile.open(archive_path) as tar:
            tar.extractall(self.root, filter="data")

    def samples(self, n):
        split_dir = self.root / "aclImdb" / self.split
        pos_files = sorted((split_dir / "pos").glob("*.txt"))
        neg_files = sorted((split_dir / "neg").glob("*.txt"))

        # Interleave pos/neg so a small `n` isn't all one label.
        interleaved = []
        for pos_file, neg_file in zip(pos_files, neg_files):
            interleaved.append((pos_file, "positive"))
            interleaved.append((neg_file, "negative"))

        for path, label in interleaved[:n]:
            text = path.read_text(encoding="utf-8")
            yield text, label


DATASETS.register("IMDB", implemented=True)(IMDBDataset)
