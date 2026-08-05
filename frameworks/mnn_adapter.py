"""MNN framework adapter -- written to the documented conversion + inference
API, NOT execution-verified in this environment.

The standard PyTorch -> MNN path: export the architecture's PyTorch module
to ONNX (the same export step frameworks/tensorrt_adapter.py and
frameworks/onnx_runtime_adapter.py use), then convert ONNX -> MNN's own
`.mnn` format via the `mnnconvert` CLI tool bundled in the `mnn` PyPI
package (`mnnconvert --framework ONNX --modelFile model.onnx --MNNModel
model.mnn`). Inference then goes through `MNN.Interpreter` -- create a
session, copy input data into the session's input tensor, `runSession`,
read the output tensor back.

DELIBERATELY NOT RUN HERE, and this needs a stronger warning than the
usual "not execution-verified" note on ExecuTorch/TVM/NCNN: directly
testing `mnnconvert.exe --help` in this environment (mnn==3.6.1, Windows,
Python 3.14) triggered an unprompted, undisclosed `pip install
aliyun-log-python-sdk` (Alibaba Cloud's logging SDK) plus a chain of
unrelated network dependencies -- with no --help output produced at all.
That's not normal behavior for a --help flag under any framing. Nothing
from that install chain actually completed in this environment (a
downstream `head` cut the pipe before it finished, confirmed no packages
landed), and the `mnn` package has been uninstalled here as a precaution.

Before ever actually running mnnconvert (from this adapter or otherwise):
re-verify this behavior is still present in whatever mnn version you're
on, understand why it happens, and decide deliberately whether that's
acceptable for your environment -- don't assume it's been fixed or was a
one-off. This adapter's conversion step is written to MNN's documented
CLI interface and left unexecuted pending that decision.

mnn/torch are imported lazily so this module still registers cleanly --
and shows up correctly in `python main.py --list` -- on a machine without
mnn installed.
"""
import struct
import subprocess
import tempfile
from pathlib import Path

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class MNNModel:
    def __init__(self, interpreter, session):
        self.interpreter = interpreter
        self.session = session


class MNNAdapter(FrameworkAdapter):
    def __init__(self):
        import MNN  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            onnx_path = tmp_dir / "model.onnx"
            mnn_path = tmp_dir / "model.mnn"

            torch.onnx.export(
                torch_model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=13,
            )

            # See module docstring: this subprocess call is written to
            # MNN's documented CLI interface but has never actually been
            # run in this project, pending a deliberate decision about
            # mnnconvert's undisclosed network-install behavior on
            # --help. Do not remove this comment when you flip that
            # decision -- re-read the docstring first.
            subprocess.run(
                [
                    "mnnconvert",
                    "--framework", "ONNX",
                    "--modelFile", str(onnx_path),
                    "--MNNModel", str(mnn_path),
                ],
                check=True,
            )

            import MNN

            interpreter = MNN.Interpreter(str(mnn_path))
            session = interpreter.createSession()

        return MNNModel(interpreter, session)

    def predict(self, model, input_tensor):
        import MNN
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        input_mnn_tensor = model.interpreter.getSessionInput(model.session)
        tmp_input = MNN.Tensor(
            array.shape, MNN.Halide_Type_Float, array, MNN.Tensor_DimensionType_Caffe
        )
        input_mnn_tensor.copyFrom(tmp_input)

        model.interpreter.runSession(model.session)

        output_mnn_tensor = model.interpreter.getSessionOutput(model.session)
        tmp_output = MNN.Tensor(
            output_mnn_tensor.getShape(),
            MNN.Halide_Type_Float,
            np.zeros(output_mnn_tensor.getShape(), dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        output_mnn_tensor.copyToHostTensor(tmp_output)
        return torch.from_numpy(tmp_output.getNumpyData().copy())

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
    "MNN", implemented=True, organization="Alibaba", platforms=["android", "ios", "ipados"]
)
def build_mnn_adapter(**kwargs):
    return MNNAdapter()
