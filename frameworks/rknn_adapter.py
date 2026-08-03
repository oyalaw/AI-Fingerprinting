"""RKNN (Rockchip Neural Network) framework adapter -- written to the
documented `rknn-toolkit2` API, NOT execution-verified in this
environment.

`rknn-toolkit2` (PyPI) exports `rknn.api.RKNN`, Rockchip's own
programmatic conversion + runtime class: `rknn.load_onnx(...)` (same ONNX
export step frameworks/tensorrt_adapter.py uses), `rknn.build(...)`,
`rknn.export_rknn(...)` produce a `.rknn` model; `rknn.init_runtime()`
with no `target=` argument runs it through RKNN's own x86 CPU simulator
(no real Rockchip NPU hardware needed for this simulator path -- only for
`target=` pointing at real hardware over ADB), and `rknn.inference(...)`
runs it.

Confirmed here: `pip index versions rknn-toolkit2` returns no matching
distribution for this project's Python version/platform -- Rockchip only
publishes wheels for specific Linux + Python combinations, not this
machine's. Treat this the same as frameworks/executorch_adapter.py --
written to the documented API, not verified, re-check against the
current rknn-toolkit2 docs before relying on it.

rknn/torch are imported lazily so this module still registers cleanly --
and shows up correctly in `python main.py --list` -- on a machine
without rknn-toolkit2 installed.
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


class RKNNModel:
    def __init__(self, rknn):
        self.rknn = rknn


class RKNNAdapter(FrameworkAdapter):
    def __init__(self):
        from rknn.api import RKNN  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry):
        import torch
        from rknn.api import RKNN

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

            rknn = RKNN()
            rknn.config(target_platform="rk3588")
            ret = rknn.load_onnx(model=str(onnx_path))
            if ret != 0:
                raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"RKNN build failed with code {ret}")

        # No target= -> RKNN's host-side x86 CPU simulator, not real hardware.
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime (simulator) failed with code {ret}")

        return RKNNModel(rknn)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        outputs = model.rknn.inference(inputs=[np.ascontiguousarray(array)])
        return torch.from_numpy(np.asarray(outputs[0]))

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


@FRAMEWORKS.register("RKNN", implemented=True, organization="Rockchip", platforms=["linux"])
def build_rknn_adapter(**kwargs):
    return RKNNAdapter()
