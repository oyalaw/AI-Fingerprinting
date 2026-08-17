"""TensorFlow framework adapter -- written to the documented conversion
API. The original wheel-availability blocker is now resolved; retrying
surfaced two further, unrelated bugs in the conversion library underneath
it, confirmed by actually running the adapter rather than left as a
prediction.

There's no direct PyTorch -> TensorFlow path from TensorFlow itself, same
situation TensorRT/OpenVINO/TFLite are in with their own converters. The
current best-maintained community route is: export the architecture's
PyTorch module to ONNX (same export step frameworks/tensorrt_adapter.py
and frameworks/onnx_runtime_adapter.py use), then convert ONNX -> a
TensorFlow SavedModel with `onnx2tf` (actively maintained; the older
`onnx-tf` hasn't kept pace with recent ONNX opsets and is effectively
unmaintained). Inference then runs through `tf.saved_model.load(...)`'s
default serving signature.

Originally confirmed blocked: `pip index versions tensorflow` returned no
matching distribution for this project's Python version (3.14, on this
project's earlier Windows dev machine). Re-checked directly on this
project's move to Ubuntu, on Python 3.13: `pip install tensorflow`
succeeds cleanly now -- TensorFlow 2.21.0 ships a real `cp313` wheel (this
looks like a Python-version-ceiling fix that's landed since the original
check, not something OS-specific; TensorFlow 2.21.0 also ships a `cp313`
`win_amd64` wheel, so this was likely never Windows-specific to begin
with). `import tensorflow` works, correctly falls back to CPU (no GPU on
this machine), confirmed with a real `tf.constant(...)` op.

Actually running this adapter through `python main.py` (previously
impossible) surfaced two further, real bugs, both in `onnx2tf` itself, not
this adapter's own code:

1. `onnx2tf`'s own PyPI metadata under-declares its dependencies --
   `import onnx2tf` fails in stages needing `tf_keras`, `onnx_graphsurgeon`,
   `ai_edge_litert`, then `sng4onnx` in turn, none pulled in automatically
   by `pip install onnx2tf`. Installed all four directly.

2. With imports fixed, `onnx2tf.convert(...)` itself then fails:
   `ValueError: This file contains pickled (object) data. ... use
   allow_pickle=`. Traced directly: `onnx2tf` auto-detects an image-shaped
   input (NHWC, 3 channels, confirmed this project's NCHW ResNet18 export
   gets there after onnx2tf's own internal transpose) and downloads a
   bundled calibration test image via `np.load(f)` with no
   `allow_pickle=True` -- current numpy (2.x) refuses to unpickle object
   arrays by default for security reasons the version this file predates
   didn't enforce. The pinned `onnx2tf==1.28.8` this project's dependency
   resolution settles on (given `tensorflow==2.21.0` already installed)
   has this bug; the latest release (2.6.8) looks likely to have moved
   past it, but pins `setuptools==81.0.0` exactly -- directly conflicting
   with this project's own `setuptools==80.9.0` pin (needed elsewhere for
   `pkg_resources` availability, see cv_frameworks/mmdetection_adapter.py's
   docstring), so upgrading isn't a clean fix without risking a regression
   to every other adapter that pin protects. Not pursued further --
   confirmed no side effects (setuptools pin unchanged, no partial state
   left behind). Revisit once a released `onnx2tf` fixes this without
   raising its own `setuptools` floor past 80.9.0.

Still treat this the same as frameworks/executorch_adapter.py and
frameworks/tvm_adapter.py: written to the documented API, not verified
end-to-end -- but the reason has moved from "TensorFlow doesn't install
here" to "TensorFlow installs and runs fine, onnx2tf's own bugs are the
remaining wall."

onnx2tf/tensorflow/torch are all imported lazily so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
on a machine without them installed.
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


class TensorFlowModel:
    def __init__(self, serving_fn, output_key):
        self.serving_fn = serving_fn
        self.output_key = output_key


class TensorFlowAdapter(FrameworkAdapter):
    def __init__(self):
        import tensorflow  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import onnx2tf
        import tensorflow as tf
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            onnx_path = tmp_dir / "model.onnx"
            saved_model_dir = tmp_dir / "saved_model"

            with torch.no_grad():
                torch.onnx.export(
                    torch_model,
                    dummy_input,
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                )

            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                non_verbose=True,
            )
            saved_model = tf.saved_model.load(str(saved_model_dir))

        serving_fn = saved_model.signatures["serving_default"]
        output_key = list(serving_fn.structured_outputs.keys())[0]
        return TensorFlowModel(serving_fn, output_key)

    def predict(self, model, input_tensor):
        import tensorflow as tf
        import torch

        array = input_tensor.detach().cpu().numpy()
        if array.ndim == 3:
            array = array[None, ...]

        output = model.serving_fn(tf.constant(array))[model.output_key]
        return torch.from_numpy(output.numpy())

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
    "TensorFlow", implemented=True, organization="Google", platforms=["windows", "linux", "macos", "jetson"]
)
def build_tensorflow_adapter(**kwargs):
    return TensorFlowAdapter()
