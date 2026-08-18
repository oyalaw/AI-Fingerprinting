"""TensorFlow framework adapter -- real implementation, verified end-to-end.
The original wheel-availability blocker is resolved, and the real
conversion-library bugs found underneath it (three of them, layered) are
now all resolved too, confirmed by actually running the adapter rather
than left as a prediction.

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
impossible) surfaced further real bugs, all in `onnx2tf` itself, not
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
   to every other adapter that pin protects.

   **Found a single fix that resolves all three layers at once, without
   touching the setuptools pin**: the buggy `download_test_image_data()`
   path only triggers when `onnx2tf` sees an input it recognizes as
   image-shaped -- specifically a concrete, small batch dimension *and*
   channels-last (`shape[-1] == 3`), which only happens after `onnx2tf`'s
   own internal NCHW->NHWC transpose. First tried forcing the batch
   dimension dynamic (`dynamic_axes=...`) to dodge that check -- it
   worked for the pickle bug specifically, but surfaced two *more* real
   bugs in turn: a channel-order mismatch (this adapter's own `predict()`
   still passed NCHW arrays, but the NHWC-converted model now expected
   NHWC) and a genuine shape-mismatch bug inside `onnx2tf`'s own graph,
   specifically triggered by the dynamic axis. The two fixes were in
   tension -- fixing one re-triggered the other.

   The actual fix needed is simpler and addresses the root of all three
   at once: `onnx2tf.convert(..., keep_ncw_or_nchw_or_ncdhw_input_names=
   ["input"])`, a real, documented `onnx2tf` parameter that keeps the
   converted model in the *original* NCHW layout instead of converting to
   NHWC at all. With no NHWC transpose happening, `onnx2tf` never
   recognizes the input as image-shaped in the first place -- the
   calibration-image download (and its pickle bug) is never reached, no
   dynamic axis is needed (so no shape-mismatch bug), and the converted
   model's own `structured_input_signature` now correctly reports
   `(1, 3, 32, 32)` (NCHW), matching this adapter's `predict()` exactly
   with zero changes needed there. Confirmed directly with a real
   standalone script: `onnx2tf.convert()` completes, and the converted
   model's output matches the real PyTorch reference exactly
   (`np.allclose(..., atol=1e-3)` true) for real ResNet18, not a toy
   module.

**Verified end-to-end**: `load_model()`/`predict()` below now produce a
real, correct TensorFlow SavedModel inference path -- confirmed matching
PyTorch's own output for the same input, and confirmed with a real
`python main.py --role server` / `--role client` roundtrip (not just a
direct adapter call) serving real predictions over the wire, the same
verification bar frameworks/onnx_runtime_adapter.py and
frameworks/openvino_adapter.py were held to. (One red herring along the
way: a `--role server` run looked hung for several minutes when its
stdout was redirected to a log file for polling -- Python block-buffers
stdout when it isn't a TTY, so "Server listening" sat unflushed in the
buffer while the process was already correctly blocked in `accept()`,
confirmed via `/proc/<pid>/wchan` reading `inet_csk_accept` while the log
looked stalled. Not a real bug, just a testing artifact.)

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
                # Keeps the model's input in this project's own NCHW layout
                # instead of onnx2tf's NHWC-by-default conversion -- see this
                # module's docstring: this single flag is what avoids all
                # three of the real onnx2tf bugs found during investigation,
                # together, without touching predict() or the export step.
                keep_ncw_or_nchw_or_ncdhw_input_names=["input"],
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
