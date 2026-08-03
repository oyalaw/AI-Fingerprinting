"""PyTorch framework adapter — the one fully-implemented deployment
framework this pass. Tensors cross the wire as raw bytes: a small binary
header (dtype code + ndim + shape) followed by the flat buffer, so no
pickle is needed (safer, and closer to how a real serving protocol frames
tensors than pickling arbitrary Python objects).

torch/numpy are imported lazily inside methods, not at module scope, so
`python main.py --list` can enumerate this registration without torch
installed -- only actually building/using the adapter needs it.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64}


class PyTorchAdapter(FrameworkAdapter):
    def __init__(self):
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, architecture_entry):
        model = architecture_entry.build(self)
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
        np_dtype = _np_dtype_map()
        tensor = tensor.detach().cpu()
        dtype_name = str(tensor.dtype).replace("torch.", "")
        dtype_code = _DTYPE_NAME_TO_CODE.get(dtype_name, 0)
        array = tensor.numpy().astype(np_dtype[dtype_code])
        shape = array.shape
        header = struct.pack(">BB", dtype_code, len(shape)) + struct.pack(f">{len(shape)}I", *shape)
        return header + array.tobytes()

    def deserialize(self, data: bytes):
        import numpy as np
        import torch

        np_dtype = _np_dtype_map()
        dtype_code, ndim = struct.unpack(">BB", data[:2])
        offset = 2
        shape = struct.unpack(f">{ndim}I", data[offset:offset + 4 * ndim])
        offset += 4 * ndim
        array = np.frombuffer(data[offset:], dtype=np_dtype[dtype_code]).reshape(shape)
        return torch.from_numpy(array.copy())


@FRAMEWORKS.register("PyTorch", implemented=True, organization="Meta", platforms=["windows", "linux", "macos", "jetson"])
def build_pytorch_adapter(**kwargs):
    return PyTorchAdapter()
