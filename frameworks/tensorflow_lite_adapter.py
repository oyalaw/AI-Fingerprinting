"""TensorFlow Lite (LiteRT) framework adapter -- real implementation.

There's no official direct PyTorch -> TFLite path from TensorFlow itself
(unlike TensorRT's ONNX import or OpenVINO's convert_model). The current
Google-supported route is `litert-torch` (formerly `ai-edge-torch`, which
is now a deprecated compatibility alias for the same API): it converts a
PyTorch module straight to a `.tflite` flatbuffer. Inference then runs
through the standard TFLite interpreter API -- preferring the lightweight,
TensorFlow-independent `ai-edge-litert` runtime, falling back to
`tensorflow.lite.Interpreter` if only full TensorFlow is installed.

litert_torch/ai_edge_torch/ai_edge_litert/tensorflow/torch are all imported
lazily so this module still registers cleanly -- and shows up correctly in
`python main.py --list` -- on a machine with none of them installed.
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


def _load_converter():
    try:
        import litert_torch

        return litert_torch.convert
    except ImportError:
        pass
    try:
        import ai_edge_torch  # deprecated alias for litert_torch, same API

        # Some installed versions of the deprecated `ai-edge-torch` package are
        # just a deprecation-notice stub with no real `convert` -- treat a
        # missing attribute the same as the package not being installed.
        return ai_edge_torch.convert
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "TensorFlow Lite conversion needs a working `litert-torch` install "
            "(or its deprecated alias `ai-edge-torch`), which in turn needs "
            "`tensorflow` -- neither has a wheel for every Python version yet "
            "(e.g. brand-new Python releases). Use a Python version with a "
            "`tensorflow` wheel available."
        ) from exc


def _load_interpreter_cls():
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.lite.Interpreter
    except ImportError as exc:
        raise ImportError(
            "TensorFlow Lite inference needs either `ai-edge-litert` (lightweight, "
            "no full TensorFlow required) or `tensorflow` installed."
        ) from exc


class TFLiteModel:
    def __init__(self, interpreter, input_detail, output_detail):
        self.interpreter = interpreter
        self.input_detail = input_detail
        self.output_detail = output_detail


class TensorFlowLiteAdapter(FrameworkAdapter):
    def __init__(self):
        self._convert = _load_converter()
        self._interpreter_cls = _load_interpreter_cls()

    def load_model(self, architecture_entry, config):
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        sample_input = (torch.randn(1, *input_shape),)

        edge_model = self._convert(torch_model, sample_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tflite_path = Path(tmp_dir) / "model.tflite"
            edge_model.export(str(tflite_path))
            tflite_bytes = tflite_path.read_bytes()

        interpreter = self._interpreter_cls(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        return TFLiteModel(interpreter, input_detail, output_detail)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        model.interpreter.resize_tensor_input(model.input_detail["index"], array.shape)
        model.interpreter.allocate_tensors()
        model.interpreter.set_tensor(model.input_detail["index"], array)
        model.interpreter.invoke()
        output = model.interpreter.get_tensor(model.output_detail["index"])
        return torch.from_numpy(output)

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
    "TensorFlow Lite",
    implemented=True,
    organization="Google",
    platforms=["android", "ios", "ipados", "linux", "windows", "macos", "jetson"],
)
def build_tflite_adapter(**kwargs):
    return TensorFlowLiteAdapter()
