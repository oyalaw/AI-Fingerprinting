"""TensorFlow framework adapter -- written to the documented conversion
API, NOT execution-verified in this environment.

There's no direct PyTorch -> TensorFlow path from TensorFlow itself, same
situation TensorRT/OpenVINO/TFLite are in with their own converters. The
current best-maintained community route is: export the architecture's
PyTorch module to ONNX (same export step frameworks/tensorrt_adapter.py
and frameworks/onnx_runtime_adapter.py use), then convert ONNX -> a
TensorFlow SavedModel with `onnx2tf` (actively maintained; the older
`onnx-tf` hasn't kept pace with recent ONNX opsets and is effectively
unmaintained). Inference then runs through `tf.saved_model.load(...)`'s
default serving signature.

Confirmed blocked in this environment specifically: `pip index versions
tensorflow` returns no matching distribution for this project's Python
version, so this adapter has not been run here -- treat it the same as
frameworks/executorch_adapter.py and frameworks/tvm_adapter.py: written to
the documented API, not verified, re-check against current onnx2tf/
TensorFlow docs before relying on it.

onnx2tf/tensorflow/torch are all imported lazily so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
on a machine without them installed.
"""
import struct
import tempfile
from pathlib import Path

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class TensorFlowModel:
    def __init__(self, serving_fn, output_key):
        self.serving_fn = serving_fn
        self.output_key = output_key


class TensorFlowAdapter(FrameworkAdapter):
    def __init__(self):
        import tensorflow  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import onnx2tf
        import tensorflow as tf
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            onnx_path = tmp_dir / "model.onnx"
            saved_model_dir = tmp_dir / "saved_model"

            with torch.no_grad():
                torch.onnx.export(
                    torch_model,
                    dummy_input,
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                )

            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                non_verbose=True,
            )
            saved_model = tf.saved_model.load(str(saved_model_dir))

        serving_fn = saved_model.signatures["serving_default"]
        output_key = list(serving_fn.structured_outputs.keys())[0]
        return TensorFlowModel(serving_fn, output_key)

    def predict(self, model, input_tensor):
        import tensorflow as tf
        import torch

        array = input_tensor.detach().cpu().numpy()
        if array.ndim == 3:
            array = array[None, ...]

        output = model.serving_fn(tf.constant(array))[model.output_key]
        return torch.from_numpy(output.numpy())

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
    "TensorFlow", implemented=True, organization="Google", platforms=["windows", "linux", "macos", "jetson"]
)
def build_tensorflow_adapter(**kwargs):
    return TensorFlowAdapter()
