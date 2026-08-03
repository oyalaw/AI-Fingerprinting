"""SNPE (Qualcomm Neural Processing SDK) framework adapter -- written to
the documented SDK workflow, NOT execution-verified in this environment.

Unlike every other framework in this project, SNPE isn't `pip install`-able
at all: Qualcomm distributes it as a manually-downloaded SDK zip (requires
a free Qualcomm Developer Network account) containing its own CLI tools
under `$SNPE_ROOT/bin/`, not a Python package. The real workflow: export
the architecture's PyTorch module to ONNX (same export step
frameworks/tensorrt_adapter.py uses), convert with the SDK's
`snpe-onnx-to-dlc` CLI tool to Qualcomm's `.dlc` format, then run it with
`snpe-net-run` (CPU/DSP/GPU runtime selectable via extra flags), reading
results back from the output directory it writes.

Confirmed here: there's no PyPI package to even attempt installing --
`SNPE_ROOT` isn't set and the SDK isn't present, so `load_model` fails
immediately with a clear, actionable error rather than a confusing one
buried in a subprocess call. Treat this the same as
frameworks/executorch_adapter.py -- written to the documented workflow,
not verified, re-check against the current Qualcomm Neural Processing SDK
docs (get the SDK from https://qpm.qualcomm.com/) before relying on it.

Nothing SNPE-specific is imported at module scope, so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
without the SDK installed.
"""
import os
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


def _snpe_bin(name):
    snpe_root = os.environ.get("SNPE_ROOT")
    if not snpe_root:
        raise RuntimeError(
            "SNPE requires the Qualcomm Neural Processing SDK, which isn't a "
            "pip package -- download it from https://qpm.qualcomm.com/ (free "
            "Qualcomm Developer Network account required) and set SNPE_ROOT "
            "to the extracted SDK directory."
        )
    bin_path = Path(snpe_root) / "bin" / "x86_64-linux-clang" / name
    if not bin_path.exists():
        raise RuntimeError(f"Expected SNPE tool not found: {bin_path}")
    return str(bin_path)


class SNPEModel:
    def __init__(self, dlc_path, output_shape):
        self.dlc_path = dlc_path
        self.output_shape = output_shape


class SNPEAdapter(FrameworkAdapter):
    def load_model(self, architecture_entry):
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        work_dir = Path(tempfile.mkdtemp())
        onnx_path = work_dir / "model.onnx"
        dlc_path = work_dir / "model.dlc"

        with torch.no_grad():
            output_shape = tuple(torch_model(dummy_input).shape)
            torch.onnx.export(
                torch_model, dummy_input, str(onnx_path), input_names=["input"], output_names=["output"]
            )

        subprocess.run(
            [_snpe_bin("snpe-onnx-to-dlc"), "--input_network", str(onnx_path), "--output_path", str(dlc_path)],
            check=True,
        )
        return SNPEModel(dlc_path, output_shape)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]

        work_dir = Path(tempfile.mkdtemp())
        raw_path = work_dir / "input.raw"
        np.ascontiguousarray(array).tofile(raw_path)
        input_list_path = work_dir / "input_list.txt"
        input_list_path.write_text(f"input:={raw_path}\n")

        output_dir = work_dir / "output"
        subprocess.run(
            [
                _snpe_bin("snpe-net-run"),
                "--container",
                str(model.dlc_path),
                "--input_list",
                str(input_list_path),
                "--output_dir",
                str(output_dir),
            ],
            check=True,
        )
        output_raw = next(output_dir.glob("**/output.raw"))
        output_array = np.fromfile(output_raw, dtype=np.float32).reshape(model.output_shape)
        return torch.from_numpy(output_array)

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


@FRAMEWORKS.register("SNPE", implemented=True, organization="Qualcomm", platforms=["android"])
def build_snpe_adapter(**kwargs):
    return SNPEAdapter()
