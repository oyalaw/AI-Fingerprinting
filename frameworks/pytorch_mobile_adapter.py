"""PyTorch Mobile framework adapter -- real implementation.

PyTorch Mobile's real deployment target is a native Android
(`org.pytorch:pytorch_android`) or iOS (`LibTorch-Lite`) app -- out of
scope here for the same reason noted for every other mobile-native
framework in this project (README.md's Devices section): this environment
can't build or flash a native app. Unlike TFLite/ORT Mobile though,
PyTorch's own Lite Interpreter format can be produced *and loaded back*
entirely with the `torch` package already used everywhere else in this
project -- no extra dependency at all -- so this one is fully real and
fully testable end-to-end on desktop, standing in for the on-device Lite
Interpreter runtime a real mobile build would use.

Pipeline: `torch.jit.trace` the architecture's PyTorch module to
TorchScript, run Meta's real `optimize_for_mobile` pass (conv/batchnorm
fusion etc. -- the same optimization a real mobile build gets), save via
`_save_for_lite_interpreter` (the actual `.ptl` mobile format), then load
it back with `_load_for_lite_interpreter` and run inference on that
loaded module.

torch is imported lazily so this module still registers cleanly -- and
shows up correctly in `python main.py --list` -- on a machine without it
installed.
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


class PyTorchMobileModel:
    def __init__(self, lite_module):
        self.lite_module = lite_module


class PyTorchMobileAdapter(FrameworkAdapter):
    def __init__(self):
        import torch  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry):
        import torch
        from torch.jit.mobile import _load_for_lite_interpreter
        from torch.utils.mobile_optimizer import optimize_for_mobile

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with torch.no_grad():
            traced = torch.jit.trace(torch_model, dummy_input)

        try:
            optimized = optimize_for_mobile(traced)
        except RuntimeError as exc:
            # Mobile optimization (conv/batchnorm fusion etc.) needs XNNPACK,
            # which not every PyTorch build includes (e.g. some CPU-only
            # wheels). The Lite Interpreter format itself doesn't require it
            # -- fall back to saving the unoptimized traced module rather
            # than failing outright.
            import warnings

            warnings.warn(
                f"optimize_for_mobile unavailable ({exc}); saving the "
                "unoptimized traced module instead.",
                stacklevel=2,
            )
            optimized = traced

        # _save_for_lite_interpreter's docstring vaguely suggests a file-like
        # object might work, but the actual C++ binding only accepts a
        # filename string -- found by testing, not documentation.
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptl_path = Path(tmp_dir) / "model.ptl"
            optimized._save_for_lite_interpreter(str(ptl_path))
            with ptl_path.open("rb") as f:
                lite_module = _load_for_lite_interpreter(f)

        return PyTorchMobileModel(lite_module)

    def predict(self, model, input_tensor):
        import torch

        with torch.no_grad():
            batch = input_tensor
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            return model.lite_module(batch)

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
    "PyTorch Mobile", implemented=True, organization="Meta", platforms=["android", "ios", "ipados"]
)
def build_pytorch_mobile_adapter(**kwargs):
    return PyTorchMobileAdapter()
