"""ONNX Runtime framework adapter -- real implementation.

Exports the selected architecture's PyTorch module to ONNX (the same
export step frameworks/tensorrt_adapter.py uses as its own input format),
then runs it through `onnxruntime.InferenceSession`. Pure Python/C++
wheel, no special hardware or GPU required -- runs on any platform ONNX
Runtime ships a wheel for (which is most of them).

onnxruntime/torch are imported lazily so this module still registers
cleanly -- and shows up correctly in `python main.py --list` -- on a
machine without onnxruntime installed.
"""
import io
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class ONNXRuntimeModel:
    def __init__(self, session, input_name, output_name):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name


class ONNXRuntimeAdapter(FrameworkAdapter):
    def __init__(self):
        import onnxruntime  # noqa: F401 -- fail fast here if not installed

        self._onnxruntime = onnxruntime

    def load_model(self, architecture_entry, config):
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            torch_model,
            dummy_input,
            onnx_buffer,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,  # the newer dynamo exporter warns dynamic_axes isn't
            # recommended for it, and (at least on some PyTorch versions) emits
            # verbose Unicode progress output that crashes non-UTF-8 consoles
        )

        session = self._onnxruntime.InferenceSession(
            onnx_buffer.getvalue(), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return ONNXRuntimeModel(session, input_name, output_name)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        (output,) = model.session.run([model.output_name], {model.input_name: array})
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


@FRAMEWORKS.register(
    "ONNX Runtime", implemented=True, organization="Microsoft", platforms=["windows", "linux", "macos", "jetson"]
)
def build_onnx_runtime_adapter(**kwargs):
    return ONNXRuntimeAdapter()
