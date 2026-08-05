"""Edge Impulse framework adapter -- written to the documented
`edge_impulse_linux` local-runner API, NOT execution-verified in this
environment, and architecturally different from every other framework
adapter in this project.

Every other adapter in this project takes the *same* architecture (e.g.
ResNet18) already defined in PyTorch and converts/compiles it locally.
Edge Impulse doesn't support that: models are trained and built through
Edge Impulse's own cloud Studio (requires an Edge Impulse account), which
exports a target-specific `.eim` executable -- there is no local "convert
my own arbitrary PyTorch architecture" step at all, so this project's
ResNet18 has no Edge Impulse export to load. The real workflow, once you
do have an `.eim` (from Edge Impulse Studio, for a project you created
there): `edge_impulse_linux.image.ImageImpulseRunner(eim_path)` as a
context manager, `runner.init()`, then `runner.classify(image_array)` per
frame.

load_model here implements that real runner API faithfully, reading the
`.eim` path from `architecture_entry.meta.get("edge_impulse_eim_path")`
-- but no architecture in this project sets that metadata (there's
nothing to set it to), so calling this with the ResNet18 slice raises a
clear error explaining the mismatch rather than pretending to convert
something Edge Impulse has no local conversion story for. Treat this the
same as frameworks/executorch_adapter.py for the parts that would run
given a real `.eim` -- written to the documented API, not verified,
re-check against current `edge_impulse_linux` docs before relying on it.

edge_impulse_linux/torch are imported lazily so this module still
registers cleanly -- and shows up correctly in `python main.py --list` --
without edge_impulse_linux installed.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class EdgeImpulseModel:
    def __init__(self, runner):
        self.runner = runner


class EdgeImpulseAdapter(FrameworkAdapter):
    def __init__(self):
        import edge_impulse_linux  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        eim_path = architecture_entry.meta.get("edge_impulse_eim_path")
        if not eim_path:
            raise RuntimeError(
                f"Edge Impulse has no local conversion path for "
                f"'{architecture_entry.name}' -- unlike every other framework "
                f"in this project, Edge Impulse models are trained/exported "
                f"through Edge Impulse Studio (cloud, account required), not "
                f"converted locally from an existing PyTorch module. Train "
                f"this architecture in Edge Impulse Studio, download its .eim "
                f"export, and set architecture metadata "
                f"'edge_impulse_eim_path' to that file to use this adapter."
            )

        from edge_impulse_linux.image import ImageImpulseRunner

        runner = ImageImpulseRunner(eim_path)
        runner.init()
        return EdgeImpulseModel(runner)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array[0]
        array = np.transpose(array, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

        _features, _cropped, result = model.runner.classify(np.ascontiguousarray(array))
        classification = result["result"]["classification"]
        scores = np.array(list(classification.values()), dtype=np.float32)
        return torch.from_numpy(scores).unsqueeze(0)

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


@FRAMEWORKS.register("Edge Impulse", implemented=True, organization="Edge Impulse", platforms=["linux", "jetson"])
def build_edge_impulse_adapter(**kwargs):
    return EdgeImpulseAdapter()
