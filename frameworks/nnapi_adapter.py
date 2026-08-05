"""NNAPI framework adapter -- written to the documented API, NOT
execution-verified in this environment.

NNAPI (Android's Neural Networks API) has no standalone Python binding at
all -- it's a C API invoked via JNI from an Android app, or picked up as
an execution provider *by* another runtime. The real, documented Python
path to it is ONNX Runtime's own NNAPI execution provider (the same ONNX
export step frameworks/onnx_runtime_adapter.py already uses, but with
`providers=["NNAPIExecutionProvider", "CPUExecutionProvider"]`), which is
compiled into `onnxruntime-android`/ORT Mobile builds, not the plain
`onnxruntime` PyPI wheel this project already depends on for the ONNX
Runtime adapter.

Confirmed directly in this environment: `onnxruntime.get_available_providers()`
returns only `['AzureExecutionProvider', 'CPUExecutionProvider']` -- no
NNAPIExecutionProvider, exactly as expected on a non-Android ORT wheel.

The naive version of this adapter would be actively misleading: ORT's
`providers=[...]` argument is a *priority* list, and if the first entry
isn't compiled in, ORT silently falls back to the next one rather than
raising an error -- so simply requesting NNAPIExecutionProvider here would
silently run on CPUExecutionProvider instead, and the resulting traffic
would carry an "NNAPI" ground-truth label for a request that never touched
NNAPI at all. For a fingerprinting research tool where the label IS the
point, that's worse than failing outright. So this adapter checks
`"NNAPIExecutionProvider" in onnxruntime.get_available_providers()` itself
at load_model() time and raises a clear error if it's absent, the same
"check before silently degrading" pattern frameworks/mps_backend_adapter.py
uses for `torch.backends.mps.is_available()`.

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


class NNAPIModel:
    def __init__(self, session, input_name, output_name):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name


class NNAPIAdapter(FrameworkAdapter):
    def __init__(self):
        import onnxruntime  # noqa: F401 -- fail fast here if not installed

        self._onnxruntime = onnxruntime

    def load_model(self, architecture_entry, config):
        import torch

        available = self._onnxruntime.get_available_providers()
        if "NNAPIExecutionProvider" not in available:
            raise RuntimeError(
                "NNAPIExecutionProvider is not available in this onnxruntime "
                f"build (available providers here: {available}). NNAPI is an "
                "Android-only execution provider, compiled into "
                "onnxruntime-android / ORT Mobile builds, not the plain "
                "onnxruntime PyPI wheel. This is a real, expected failure on "
                "any non-Android machine -- see this module's docstring."
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
            dynamo=False,
        )

        session = self._onnxruntime.InferenceSession(
            onnx_buffer.getvalue(),
            providers=["NNAPIExecutionProvider", "CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return NNAPIModel(session, input_name, output_name)

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


@FRAMEWORKS.register("NNAPI", implemented=True, organization="Google", platforms=["android"])
def build_nnapi_adapter(**kwargs):
    return NNAPIAdapter()
