"""MPS Backend framework adapter -- real implementation.

Apple's Metal Performance Shaders backend for PyTorch (`torch.device("mps")`)
-- Apple Silicon GPU acceleration accessed through PyTorch itself, not a
separate package to install. Architecturally this is closer to TensorRT's
relationship with CUDA than to a converter framework like OpenVINO/ONNX
Runtime: it isn't compiling the model into a different format, it's
PyTorch's own execution backend targeting different hardware, so
`load_model` just moves the architecture's PyTorch module onto `mps`
instead of `cpu`/`cuda`.

`torch.backends.mps.is_available()` is checked directly in `__init__`
rather than letting a failed op raise deep inside `predict`. Verified on
this Windows machine: PyTorch's own MPS backend correctly reports
`is_built()=False` / `is_available()=False` here (Windows/Linux torch
builds don't compile the Metal backend in at all), and this adapter raises
a clear RuntimeError from that check immediately, at `.build()` time,
rather than silently falling back to CPU -- a silent fallback would defeat
the whole point of a device-specific fingerprinting label.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"torch.float32": 0, "torch.float64": 1, "torch.int64": 2}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64}


class MPSBackendAdapter(FrameworkAdapter):
    def __init__(self):
        import torch

        if not torch.backends.mps.is_available():
            reason = (
                "not built into this torch install (no Metal backend compiled in)"
                if not torch.backends.mps.is_built()
                else "no Apple GPU detected"
            )
            raise RuntimeError(
                "MPS Backend requires Apple Silicon and a torch build with Metal "
                f"support; {reason} on this machine."
            )
        self.device = torch.device("mps")

    def load_model(self, architecture_entry, config):
        model = architecture_entry.build(self, config)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, model, input_tensor):
        import torch

        with torch.no_grad():
            batch = input_tensor.to(self.device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            return model(batch).cpu()

    def serialize(self, tensor) -> bytes:
        import numpy as np

        tensor = tensor.detach().cpu()
        np_dtype_map = _np_dtype_map()
        dtype_code = _DTYPE_NAME_TO_CODE.get(str(tensor.dtype), 0)
        array = tensor.numpy().astype(np_dtype_map[dtype_code])
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


@FRAMEWORKS.register("MPS Backend", implemented=True, organization="Apple", platforms=["macos"])
def build_mps_backend_adapter(**kwargs):
    return MPSBackendAdapter()
