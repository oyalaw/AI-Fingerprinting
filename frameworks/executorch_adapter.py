"""ExecuTorch framework adapter -- real implementation.

ExecuTorch is Meta's newer, actively-developed successor to PyTorch Mobile
(see frameworks/pytorch_mobile_adapter.py's docstring: the framework table
itself notes PyTorch Mobile is "being replaced by ExecuTorch"). Its real
deployment target is a native Android/iOS/embedded runtime -- out of scope
here for the same reason as every other mobile-native framework in this
project. What's implemented here is the real, currently-documented desktop
export + host-side execution path:

  1. `torch.export.export` the architecture's PyTorch module to an
     ExportedProgram (torch's ahead-of-time graph capture).
  2. `executorch.exir.to_edge` lowers that to the Edge dialect.
  3. `.to_executorch()` produces the actual `.pte` program bytes -- the
     same file format a real on-device ExecuTorch runtime loads.
  4. Loaded back and run via ExecuTorch's own portable host runtime
     (`executorch.extension.pybindings.portable_lib._load_for_executorch`),
     which is ExecuTorch's documented way to validate a `.pte` on desktop
     without a real device -- not a shortcut invented for this project.

Environment note: unlike TensorRT (blocked by lacking an NVIDIA GPU), this
one is blocked by dependency availability, not hardware -- `pip install
executorch` currently has no wheel for Python 3.14 (confirmed: `Could not
find a version that satisfies the requirement executorch (from versions:
none)`), so this couldn't be run end-to-end in this project's own dev
environment. The code below follows ExecuTorch's current documented
export API as closely as this project's other adapters follow theirs; it
has not been execution-verified the way the PyTorch/OpenVINO/ONNX
Runtime/ONNX Runtime Mobile/PyTorch Mobile adapters were. Re-check this
against ExecuTorch's docs when a compatible Python is available, since
this API has changed across ExecuTorch releases before.

torch/executorch are imported lazily so this module still registers
cleanly -- and shows up correctly in `python main.py --list` -- on a
machine without them installed.
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


class ExecuTorchModel:
    def __init__(self, et_module):
        self.et_module = et_module


class ExecuTorchAdapter(FrameworkAdapter):
    def __init__(self):
        import executorch  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import torch
        from executorch.exir import to_edge
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        from torch.export import export

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        example_inputs = (torch.randn(1, *input_shape),)

        exported_program = export(torch_model, example_inputs)
        edge_program = to_edge(exported_program)
        executorch_program = edge_program.to_executorch()

        # The pybindings loader only accepts a filename, not an in-memory
        # buffer (same constraint hit with PyTorch Mobile's Lite
        # Interpreter and TFLite's converter -- write, then load back).
        with tempfile.TemporaryDirectory() as tmp_dir:
            pte_path = Path(tmp_dir) / "model.pte"
            pte_path.write_bytes(executorch_program.buffer)
            et_module = _load_for_executorch(str(pte_path))

        return ExecuTorchModel(et_module)

    def predict(self, model, input_tensor):
        batch = input_tensor
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        outputs = model.et_module.forward([batch])
        return outputs[0]

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
    "ExecuTorch", implemented=True, organization="Meta", platforms=["android", "ios", "ipados", "macos"]
)
def build_executorch_adapter(**kwargs):
    return ExecuTorchAdapter()
