"""Arm NN framework adapter -- written to the documented `pyarmnn` API,
NOT execution-verified in this environment.

Arm NN's Python bindings (`pyarmnn`) aren't a normal PyPI package for most
platforms -- Arm ships prebuilt wheels per-release on GitHub
(github.com/ARM-software/armnn/releases) matched to a specific Arm NN
build, or you build them from source. The real workflow: export the
architecture's PyTorch module to ONNX (same export step
frameworks/tensorrt_adapter.py uses), parse it with
`pyarmnn.ICreateOnnxParser().CreateNetworkFromBinaryFile(...)`, optimize
for a backend (`CpuAcc`/`GpuAcc`/`CpuRef`) with `pyarmnn.Optimize(...)`,
load it into an `IRuntime`, and run it via `EnqueueWorkload`.

Confirmed here: `pyarmnn` isn't installable via plain `pip install` on
this Windows x86_64 machine -- it targets Arm platforms (Android, Linux
on Arm, Jetson's Arm CPU). Treat this the same as
frameworks/executorch_adapter.py -- written to the documented API, not
verified, re-check against the current Arm NN release you're targeting
(the API has changed across major versions) before relying on it.

pyarmnn/torch are imported lazily so this module still registers cleanly
-- and shows up correctly in `python main.py --list` -- on a machine
without pyarmnn installed.
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


class ArmNNModel:
    def __init__(self, runtime, net_id, input_binding, output_binding):
        self.runtime = runtime
        self.net_id = net_id
        self.input_binding = input_binding
        self.output_binding = output_binding


class ArmNNAdapter(FrameworkAdapter):
    def __init__(self):
        import pyarmnn  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry):
        import pyarmnn as ann
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    torch_model, dummy_input, str(onnx_path), input_names=["input"], output_names=["output"]
                )

            parser = ann.ICreateOnnxParser()
            network = parser.CreateNetworkFromBinaryFile(str(onnx_path))

        input_binding = parser.GetNetworkInputBindingInfo("input")
        output_binding = parser.GetNetworkOutputBindingInfo("output")

        preferred_backends = [ann.BackendId("CpuAcc"), ann.BackendId("CpuRef")]
        runtime = ann.IRuntime(ann.CreationOptions())
        opt_network, _messages = ann.Optimize(
            network, preferred_backends, runtime.GetDeviceSpec(), ann.OptimizerOptions()
        )
        net_id, _messages = runtime.LoadNetwork(opt_network)

        return ArmNNModel(runtime, net_id, input_binding, output_binding)

    def predict(self, model, input_tensor):
        import numpy as np
        import pyarmnn as ann
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        input_tensors = ann.make_input_tensors([model.input_binding], [np.ascontiguousarray(array)])
        out_shape = model.output_binding[1].GetShape()
        output_tensors = ann.make_output_tensors([model.output_binding])

        model.runtime.EnqueueWorkload(model.net_id, input_tensors, output_tensors)
        output = ann.workload_tensors_to_ndarray(output_tensors)[0]
        return torch.from_numpy(output.reshape(tuple(out_shape)))

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


@FRAMEWORKS.register("Arm NN", implemented=True, organization="Arm", platforms=["android", "linux", "jetson"])
def build_arm_nn_adapter(**kwargs):
    return ArmNNAdapter()
