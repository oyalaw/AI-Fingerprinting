"""ONNX Runtime Mobile framework adapter -- real implementation.

ORT Mobile's actual deployment target is a native Android (onnxruntime-
android AAR) or iOS (onnxruntime-objc pod) app -- there's no Python runtime
on-device, so real hardware execution via NNAPI/CoreML execution providers
is out of scope here for the same reason noted in README.md: Android/
iPhone/iPad are thin network clients in this project, not something this
environment can build/flash a native app onto.

What *is* real and fully testable on desktop is ORT Mobile's actual model
*preparation* pipeline, which every ORT Mobile deployment goes through
before it ever reaches a device: export to ONNX (the same step
frameworks/onnx_runtime_adapter.py uses), then convert to the mobile-
optimized ORT flatbuffer format via `onnxruntime.tools.
convert_onnx_models_to_ort` (the real, documented public API backing the
`python -m onnxruntime.tools.convert_onnx_models_to_ort` CLI). The
resulting .ort model is then loaded back through `onnxruntime.
InferenceSession`, which accepts ORT-format models directly -- so this
validates the exact artifact a mobile app would ship, running on the
CPUExecutionProvider as a stand-in for the on-device NNAPI/CoreML EP a real
Android/iOS build would select.

onnxruntime/torch are imported lazily so this module still registers
cleanly -- and shows up correctly in `python main.py --list` -- on a
machine without onnxruntime installed.
"""
import io
import struct
import tempfile
from pathlib import Path

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class ONNXRuntimeMobileModel:
    def __init__(self, session, input_name, output_name):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name


class ONNXRuntimeMobileAdapter(FrameworkAdapter):
    def __init__(self):
        import onnxruntime  # noqa: F401 -- fail fast here if not installed

        self._onnxruntime = onnxruntime

    def load_model(self, architecture_entry, config):
        import torch
        from onnxruntime.tools.convert_onnx_models_to_ort import (
            OptimizationStyle,
            convert_onnx_models_to_ort,
        )

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
            dynamo=False,  # see frameworks/onnx_runtime_adapter.py for why
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            onnx_path = tmp_dir / "model.onnx"
            onnx_path.write_bytes(onnx_buffer.getvalue())

            # OptimizationStyle.Fixed: optimize assuming the input shape won't
            # change at runtime -- the standard choice for a mobile app serving
            # one known workload shape, matching how this testbed always calls
            # predict() with the architecture's fixed input_shape.
            convert_onnx_models_to_ort(
                onnx_path,
                output_dir=tmp_dir,
                optimization_styles=[OptimizationStyle.Fixed],
            )
            ort_path = tmp_dir / "model.ort"
            ort_bytes = ort_path.read_bytes()

        session = self._onnxruntime.InferenceSession(ort_bytes, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return ONNXRuntimeMobileModel(session, input_name, output_name)

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
    "ONNX Runtime Mobile",
    implemented=True,
    organization="Microsoft",
    platforms=["android", "ios", "ipados"],
)
def build_onnx_runtime_mobile_adapter(**kwargs):
    return ONNXRuntimeMobileAdapter()
