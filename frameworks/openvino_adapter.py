"""OpenVINO framework adapter -- real implementation.

Converts the selected architecture's PyTorch module directly to OpenVINO's
IR format via `openvino.convert_model` (OpenVINO 2023.0+'s Model
Conversion API -- no separate ONNX export or Model Optimizer CLI step
needed, unlike the older OpenVINO workflow), compiles it for a target
device, and runs inference through the compiled model.

Unlike TensorRT, OpenVINO's `CPU` plugin runs on any x86/ARM machine --
Intel-optimized, not Intel-exclusive -- so no special hardware is required
for the default device used here. `GPU`/`NPU` can target an Intel
integrated GPU/NPU if present, by passing device="GPU" to the adapter.

openvino/torch are imported lazily so this module still registers cleanly
-- and shows up correctly in `python main.py --list` -- on a machine
without openvino installed.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class OpenVINOModel:
    def __init__(self, compiled_model, output_port):
        self.compiled_model = compiled_model
        self.output_port = output_port


class OpenVINOAdapter(FrameworkAdapter):
    def __init__(self, device="CPU"):
        import openvino as ov

        self._ov = ov
        self._core = ov.Core()
        self._device = device

    def load_model(self, architecture_entry):
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        ov_model = self._ov.convert_model(torch_model, example_input=dummy_input)
        compiled_model = self._core.compile_model(ov_model, self._device)
        output_port = compiled_model.output(0)

        return OpenVINOModel(compiled_model, output_port)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        result = model.compiled_model(array)
        output = result[model.output_port]
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


@FRAMEWORKS.register("OpenVINO", implemented=True, organization="Intel", platforms=["windows", "linux", "macos"])
def build_openvino_adapter(**kwargs):
    return OpenVINOAdapter()
