"""TensorFlow Lite Micro framework adapter -- written to the documented
Python runtime API, NOT execution-verified in this environment.

TFLite Micro targets microcontrollers with no OS -- there's no pip package
at all (unlike ExecuTorch/TVM/TensorFlow, which at least publish PyPI
packages that simply lack a wheel for every Python version here). The
real workflow: build/convert to a `.tflite` flatbuffer (the same target
format frameworks/tensorflow_lite_adapter.py already produces via
litert-torch), then simulate it on-host through TFLite Micro's own Python
interpreter binding (`tflite_micro.python.tflite_micro.runtime.Interpreter`
in the github.com/tensorflow/tflite-micro repo) before flashing it to a
real microcontroller target -- this project doesn't include a
microcontroller in its device list, so that final on-device step is out
of scope even in principle, same boundary as this project's NNAPI/SNPE/
RKNN mobile-native adapters.

Confirmed here: `pip index versions tflite-micro` returns no matching
distribution -- there's no PyPI install at all, only a from-source Bazel
build against the tflite-micro GitHub repo. Treat this the same as
frameworks/executorch_adapter.py -- written to the documented API, not
verified, re-check against the current tflite-micro repo before relying
on it, and expect to build it from source rather than `pip install`.

tflite_micro/litert_torch/torch are all imported lazily so this module
still registers cleanly -- and shows up correctly in `python main.py
--list` -- on a machine without any of them installed.
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
            "TFLite Micro's model-prep step needs a `.tflite` model, which "
            "needs `litert-torch` (or its deprecated alias `ai-edge-torch`), "
            "which in turn needs `tensorflow` -- neither has a wheel for "
            "every Python version yet."
        ) from exc


class TFLiteMicroModel:
    def __init__(self, interpreter):
        self.interpreter = interpreter


class TensorFlowLiteMicroAdapter(FrameworkAdapter):
    def __init__(self):
        import tflite_micro  # noqa: F401 -- fail fast here if not installed

        self._convert = _load_tflite_converter()

    def load_model(self, architecture_entry, config):
        import torch
        from tflite_micro.python.tflite_micro import runtime

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        sample_input = (torch.randn(1, *input_shape),)
        edge_model = self._convert(torch_model, sample_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tflite_path = Path(tmp_dir) / "model.tflite"
            edge_model.export(str(tflite_path))
            tflite_bytes = tflite_path.read_bytes()

        interpreter = runtime.Interpreter.from_bytes(tflite_bytes)
        return TFLiteMicroModel(interpreter)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        model.interpreter.set_input(np.ascontiguousarray(array), 0)
        model.interpreter.invoke()
        output = model.interpreter.get_output(0)
        return torch.from_numpy(np.asarray(output))

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


@FRAMEWORKS.register("TensorFlow Lite Micro", implemented=True, organization="Google", platforms=["android"])
def build_tflite_micro_adapter(**kwargs):
    return TensorFlowLiteMicroAdapter()
