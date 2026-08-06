"""COCO (2017 val split, object detection instances) -- downloaded
directly from the official mirror with the standard library, same "no
HuggingFace `datasets` package" reasoning as datasets/imdb.py/sst2.py.

The full annotations_trainval2017.zip bundles instances/captions/
person_keypoints for BOTH train and val splits (252MB total) -- COCO
doesn't offer per-file downloads, only this one bundle. Rather than
extract the whole thing, only `annotations/instances_val2017.json` is
pulled out of the zip (confirmed directly by downloading and inspecting
it: top-level keys images/annotations/categories, 5000 images, 80
categories, each image entry carries its own `coco_url`). Images
themselves are NOT bulk-downloaded either -- the official val2017.zip of
all 5000 images is ~1GB; instead, samples() lazily fetches only the
individual JPEGs it actually needs, one HTTP GET per image via that
image's own `coco_url`, ~100-200KB each, cached to disk after first fetch.

true_label is the most common category name among an image's ground-truth
annotations -- one representative label, the same shape every other
dataset here yields (CIFAR10/IMDB/SST2 all give a single string label
too), not the full multi-object annotation list a real detector would
train against.
"""
import collections
import json
import pathlib
import urllib.request
import zipfile

import numpy as np

from core.registry import DATASETS
from datasets.base import Dataset

_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"


class COCODataset(Dataset):
    def __init__(self, root="./data/coco"):
        self.root = pathlib.Path(root)
        self._ensure_annotations()
        self._index = self._build_index()

    def _ensure_annotations(self):
        self.annotations_path = self.root / "instances_val2017.json"
        if self.annotations_path.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / "annotations_trainval2017.zip"
        if not archive_path.exists():
            urllib.request.urlretrieve(_ANNOTATIONS_URL, archive_path)
        with zipfile.ZipFile(archive_path) as z:
            with z.open(_ANNOTATIONS_MEMBER) as src, self.annotations_path.open("wb") as dst:
                dst.write(src.read())

    def _build_index(self):
        with self.annotations_path.open(encoding="utf-8") as f:
            data = json.load(f)

        category_names = {c["id"]: c["name"] for c in data["categories"]}
        categories_by_image = collections.defaultdict(list)
        for ann in data["annotations"]:
            categories_by_image[ann["image_id"]].append(category_names[ann["category_id"]])

        images = []
        for image_info in data["images"]:
            names = categories_by_image.get(image_info["id"])
            if not names:
                continue
            label = collections.Counter(names).most_common(1)[0][0]
            images.append((image_info["file_name"], image_info["coco_url"], label))
        return images

    def _fetch_image(self, file_name, coco_url):
        image_path = self.root / "images" / file_name
        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(coco_url, image_path)
        from PIL import Image

        return np.array(Image.open(image_path).convert("RGB"))

    def samples(self, n):
        for file_name, coco_url, label in self._index[:n]:
            array = self._fetch_image(file_name, coco_url)
            yield array, label


DATASETS.register("COCO", implemented=True)(COCODataset)
