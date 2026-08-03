"""CoreML framework adapter -- real implementation.

Converts the selected architecture's PyTorch module to an actual Core ML
model via `coremltools.convert` (the standard PyTorch->CoreML path:
`torch.jit.trace` first, coremltools doesn't accept a raw nn.Module), then
runs it through the converted `MLModel`'s own `.predict()`.

Uses `convert_to="neuralnetwork"` (CoreML's older but still fully
supported model format, serialized as plain protobuf) rather than the
modern default `"mlprogram"`: verified directly in this environment that
`mlprogram` conversion requires a native `libmilstoragepython` blob-writer
extension coremltools only ships for macOS, so it fails even to *convert*
here with `RuntimeError: BlobWriter not loaded`. `neuralnetwork` has no
such dependency and converts successfully on any platform coremltools
installs on.

Execution is a separate story from conversion, and is genuinely
macOS/iOS-only -- verified directly: `MLModel.predict()` raises
`Model prediction is only supported on macOS version 10.13 or later` on
this Windows machine. That's not this adapter papering over a gap; it's
coremltools' own real behavior, structurally the same situation as
frameworks/tensorrt_adapter.py needing an NVIDIA GPU -- conversion is
real and portable, execution needs the matching hardware/OS, and this
adapter doesn't fake around that.

torch/coremltools are imported lazily so this module still registers
cleanly -- and shows up correctly in `python main.py --list` -- on a
machine without them installed.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class CoreMLModel:
    def __init__(self, mlmodel):
        self.mlmodel = mlmodel


class CoreMLAdapter(FrameworkAdapter):
    def __init__(self):
        import coremltools  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry):
        import coremltools as ct
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        example_input = torch.randn(1, *input_shape)

        with torch.no_grad():
            traced_model = torch.jit.trace(torch_model, example_input)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            outputs=[ct.TensorType(name="output")],
            convert_to="neuralnetwork",
        )
        return CoreMLModel(mlmodel)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        output_dict = model.mlmodel.predict({"input": array})
        return torch.from_numpy(np.asarray(output_dict["output"]))

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
    "CoreML", implemented=True, organization="Apple", platforms=["ios", "ipados", "macos"]
)
def build_coreml_adapter(**kwargs):
    return CoreMLAdapter()
