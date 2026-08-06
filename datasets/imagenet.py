"""ImageNet (ILSVRC2012 validation split) -- downloaded directly from the
official image-net.org mirror, no login/registration wall encountered:
confirmed directly with a Range request that
`https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar` serves
real tar bytes (not an HTML login redirect) with no auth header at all.
That tar is ~6.3GB (50,000 images) -- rather than download it, `samples()`
opens it as an HTTP stream and reads it via `tarfile`'s streaming mode
(`mode="r|"`), which processes entries sequentially without seeking, and
simply stops (closing the connection) once it has the `n` images it
needs. Confirmed directly: fetching 3 images this way pulls only their
own bytes off the wire, not the full archive.

The validation tar's filenames (`ILSVRC2012_val_00046108.JPEG`) carry no
label -- ImageNet ships ground truth separately in a small (2.5MB) devkit
archive, `ILSVRC2012_devkit_t12.tar.gz` (downloaded and cached in full,
unlike the image tar, since it's small). Two real quirks confirmed by
downloading and inspecting it directly, documented in its own readme.txt:

  - `data/ILSVRC2012_validation_ground_truth.txt` has exactly 50,000
    lines, one integer class ID (1-1000) per line, in image-index order
    (line 1 = ILSVRC2012_val_00000001.JPEG's class).
  - `data/meta.mat` (a MATLAB file, parsed via `scipy.io.loadmat`) has a
    1860-entry `synsets` struct array (1000 low-level leaf classes + 860
    higher-level WordNet ancestor synsets), sorted by `ILSVRC2012_ID` --
    so `synsets[0:1000]` are exactly the 1000 real classes in ID order,
    each with a human-readable `words` field (e.g. "kit fox, Vulpes
    macrotis") used as this dataset's label. This ID ordering does NOT
    match the alphabetical-by-WNID ordering many ImageNet-pretrained
    models' output layers use -- irrelevant here since this project's
    architectures are random-init (no pretrained ImageNet classifier to
    align class-index-order with), but worth knowing if that ever changes.

Fetched images are cached to disk (by validation index) so a repeat run
with the same `n` doesn't re-stream the tar.
"""
import pathlib
import tarfile
import urllib.request

import numpy as np

from core.registry import DATASETS
from datasets.base import Dataset

_IMAGES_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
_DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"


class ImageNetDataset(Dataset):
    def __init__(self, root="./data/imagenet"):
        self.root = pathlib.Path(root)
        self._id_to_label, self._ground_truth_ids = self._load_devkit()

    def _load_devkit(self):
        import scipy.io

        self.root.mkdir(parents=True, exist_ok=True)
        devkit_path = self.root / "ILSVRC2012_devkit_t12.tar.gz"
        if not devkit_path.exists():
            urllib.request.urlretrieve(_DEVKIT_URL, devkit_path)

        with tarfile.open(devkit_path) as tar:
            meta_bytes = tar.extractfile("ILSVRC2012_devkit_t12/data/meta.mat").read()
            gt_text = tar.extractfile(
                "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
            ).read().decode()

        import io

        mat = scipy.io.loadmat(io.BytesIO(meta_bytes))
        synsets = mat["synsets"]
        id_to_label = {}
        for i in range(1000):  # first 1000 entries are the low-level (leaf) classes
            entry = synsets[i][0]
            ilsvrc_id = int(entry["ILSVRC2012_ID"][0][0])
            words = str(entry["words"][0])
            id_to_label[ilsvrc_id] = words

        ground_truth_ids = [int(line) for line in gt_text.strip().split("\n")]
        return id_to_label, ground_truth_ids

    def _fetch_images(self, n):
        # Confirmed directly: the val tar's *physical* entry order is not
        # the same as the validation-index order in each filename, and
        # (also confirmed directly) it's class-grouped -- the first 15
        # entries in tar order were all class 660, every one of them.
        # 50,000 images / 1000 classes = 50 images/class, so a plain
        # literal-first-N read degenerates into one repeated label for
        # any realistic N. STRIDE keeps only every Nth qualifying entry
        # while streaming, confirmed directly to produce 15/15 unique
        # labels -- the same "don't let a small n collapse to one label"
        # principle datasets/imdb.py's pos/neg interleaving already
        # applies, just via a stride instead of an explicit alternation
        # since there's no fixed 2-way split to interleave here.
        _STRIDE = 97

        cache_dir = self.root / "val_images"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cached = sorted(cache_dir.glob("ILSVRC2012_val_*.JPEG"))[:n]
        if len(cached) >= n:
            return cached[:n]

        response = urllib.request.urlopen(_IMAGES_URL, timeout=30)
        try:
            tar = tarfile.open(fileobj=response, mode="r|")
            fetched = []
            for i, member in enumerate(tar):
                if not member.isfile() or i % _STRIDE != 0:
                    continue
                data = tar.extractfile(member).read()
                path = cache_dir / member.name
                path.write_bytes(data)
                fetched.append(path)
                if len(fetched) >= n:
                    break
            return fetched
        finally:
            response.close()

    def samples(self, n):
        from PIL import Image

        for path in self._fetch_images(n):
            # "ILSVRC2012_val_00046108.JPEG" -> validation index 46108 (1-indexed)
            val_index = int(path.stem.rsplit("_", 1)[1])
            class_id = self._ground_truth_ids[val_index - 1]
            label = self._id_to_label[class_id]
            array = np.array(Image.open(path).convert("RGB"))
            yield array, label


DATASETS.register("ImageNet", implemented=True)(ImageNetDataset)
