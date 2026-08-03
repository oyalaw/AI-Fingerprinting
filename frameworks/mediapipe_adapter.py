"""MediaPipe framework adapter -- written to the documented Tasks API, NOT
execution-verified in this environment.

MediaPipe's Tasks API (`mediapipe.tasks.python.vision.ImageClassifier`)
does accept a custom, not-Google-provided model, but only as a `.tflite`
flatbuffer with TFLite Metadata (labels + tensor descriptions) embedded via
MediaPipe's own `metadata_writers.image_classifier.MetadataWriter` -- a
raw un-annotated `.tflite` file (like the one
frameworks/tensorflow_lite_adapter.py produces) isn't accepted directly.
The real pipeline: produce the same `.tflite` bytes via litert-torch (as
TensorFlowLiteAdapter does), attach metadata with MetadataWriter, then
load and run through `ImageClassifier.create_from_options(...)` /
`.classify(mp.Image(...))`.

Deliberately does NOT fall back to a different Google-provided pretrained
model just because one happens to load here -- every other adapter in
this project runs the *same* ResNet18 module, and the whole point of the
ground-truth labels is comparing that one architecture's traffic across
frameworks, so swapping in an unrelated pretrained model here would break
that comparison rather than complete it.

`mediapipe` itself installs and imports cleanly in this environment
(confirmed) -- what's blocked is the same thing blocking
frameworks/tensorflow_lite_adapter.py's underlying litert-torch/
tensorflow dependency (no wheel for this project's Python version), which
this adapter's model-prep step depends on. Treat this the same as
frameworks/executorch_adapter.py -- written to the documented API, not
verified, re-check against current MediaPipe Tasks docs before relying on
it.

mediapipe/litert_torch/torch are all imported lazily so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
on a machine without them installed.
"""
import struct
import tempfile
from pathlib import Path

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}
_NUM_CLASSES = 10  # matches this project's CIFAR10 slice


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


def _load_tflite_converter():
    try:
        import litert_torch

        return litert_torch.convert
    except ImportError:
        pass
    try:
        import ai_edge_torch

        return ai_edge_torch.convert
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "MediaPipe's custom-model path needs a `.tflite` model, which needs "
            "`litert-torch` (or its deprecated alias `ai-edge-torch`), which in "
            "turn needs `tensorflow` -- neither has a wheel for every Python "
            "version yet."
        ) from exc


class MediaPipeModel:
    def __init__(self, classifier):
        self.classifier = classifier


class MediaPipeAdapter(FrameworkAdapter):
    def __init__(self):
        import mediapipe  # noqa: F401 -- fail fast here if not installed

        self._convert = _load_tflite_converter()

    def load_model(self, architecture_entry):
        import torch
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.metadata.metadata_writers import (
            image_classifier as metadata_writer,
        )
        from mediapipe.tasks.python.metadata.metadata_writers import writer_utils

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        sample_input = (torch.randn(1, *input_shape),)
        edge_model = self._convert(torch_model, sample_input)

        labels = [f"class_{i}" for i in range(_NUM_CLASSES)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            tflite_path = tmp_dir / "model.tflite"
            edge_model.export(str(tflite_path))

            labels_path = tmp_dir / "labels.txt"
            labels_path.write_text("\n".join(labels))

            writer = metadata_writer.MetadataWriter.create(
                writer_utils.load_file(str(tflite_path)),
                input_norm_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                input_norm_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                labels=metadata_writer.Labels().add_from_file(str(labels_path)),
            )
            tflite_with_metadata, _metadata_json = writer.populate()

        options = vision.ImageClassifierOptions(
            base_options=BaseOptions(model_asset_buffer=tflite_with_metadata),
            max_results=_NUM_CLASSES,
        )
        classifier = vision.ImageClassifier.create_from_options(options)
        return MediaPipeModel(classifier)

    def predict(self, model, input_tensor):
        import mediapipe as mp
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array[0]
        # (C, H, W) normalized float -> (H, W, C) uint8 for mp.Image
        array = np.transpose(array, (1, 2, 0))
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(array))
        result = model.classifier.classify(mp_image)

        scores = np.zeros(_NUM_CLASSES, dtype=np.float32)
        for category in result.classifications[0].categories:
            index = int(category.category_name.split("_")[-1])
            scores[index] = category.score
        return torch.from_numpy(scores).unsqueeze(0)

    def serialize(self, tensor) -> bytes:
        import numpy as np

        array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        np_dtype_map = _np_dtype_map()
        dtype_code = _DTYPE_NAME_TO_CODE.get(str(array.dtype), 0)
        array = array.astype(np_dtype_map[dtype_code])
        shape = array.shape
        header = struct.pack(">BB", dtype_code, len(shape)) + struct.pack(f">{len(shape)}I", *shape)
        return header + array.tobytes()

    def deserialize(self, data: bytes):
        import numpy as np
        import torch

        np_dtype_map = _np_dtype_map()
        dtype_code, ndim = struct.unpack(">BB", data[:2])
        offset = 2
        shape = struct.unpack(f">{ndim}I", data[offset:offset + 4 * ndim])
        offset += 4 * ndim
        array = np.frombuffer(data[offset:], dtype=np_dtype_map[dtype_code]).reshape(shape)
        return torch.from_numpy(array.copy())


@FRAMEWORKS.register(
    "MediaPipe", implemented=True, organization="Google", platforms=["android", "ios", "ipados", "macos"]
)
def build_mediapipe_adapter(**kwargs):
    return MediaPipeAdapter()
