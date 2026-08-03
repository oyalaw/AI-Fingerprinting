"""NCNN framework adapter -- written to the documented conversion +
inference API, NOT execution-verified in this environment.

The standard PyTorch -> NCNN path is `pnnx` (github.com/pnnx/pnnx, also on
PyPI): `pnnx.export(model, "model.pt", inputshape="[1,3,32,32]")` traces
the module and produces `model.ncnn.param`/`model.ncnn.bin` directly
(NCNN's own two-file format, analogous to TFLite's single flatbuffer).
Inference then goes through `ncnn.Net().load_param(...)` /
`.load_model(...)` and an `ncnn.Extractor`.

Deliberately not run here: `pnnx.export` invokes a compiled binary bundled
inside the `pnnx` PyPI wheel via subprocess, rather than a pure-Python
conversion API -- the same shape of risk this project already had to
correct once for a different framework's bundled tool, so per an explicit
decision this pass, it's written to the documented API and left
unexecuted rather than run unprompted. Treat this the same as
frameworks/executorch_adapter.py -- written to the documented API, not
verified, re-check against current pnnx/ncnn docs (both are young,
fast-moving projects) before relying on it.

pnnx/ncnn/torch are all imported lazily so this module still registers
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


class NCNNModel:
    def __init__(self, net, input_name, output_name):
        self.net = net
        self.input_name = input_name
        self.output_name = output_name


class NCNNAdapter(FrameworkAdapter):
    def __init__(self):
        import ncnn  # noqa: F401 -- fail fast here if not installed
        import pnnx  # noqa: F401

    def load_model(self, architecture_entry):
        import ncnn
        import pnnx
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            pt_path = tmp_dir / "model.pt"
            with torch.no_grad():
                traced = torch.jit.trace(torch_model, dummy_input)
            traced.save(str(pt_path))

            shape_str = "[" + ",".join(str(d) for d in dummy_input.shape) + "]"
            pnnx.export(str(pt_path), inputshape=shape_str)

            param_path = tmp_dir / "model.ncnn.param"
            bin_path = tmp_dir / "model.ncnn.bin"

            net = ncnn.Net()
            net.load_param(str(param_path))
            net.load_model(str(bin_path))

        return NCNNModel(net, "in0", "out0")

    def predict(self, model, input_tensor):
        import ncnn
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 4:
            array = array[0]

        mat_in = ncnn.Mat(np.ascontiguousarray(array))
        extractor = model.net.create_extractor()
        extractor.input(model.input_name, mat_in)
        _ret, mat_out = extractor.extract(model.output_name)
        return torch.from_numpy(np.array(mat_out)).unsqueeze(0)

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


@FRAMEWORKS.register("NCNN", implemented=True, organization="Tencent", platforms=["android", "ios", "ipados", "linux"])
def build_ncnn_adapter(**kwargs):
    return NCNNAdapter()
