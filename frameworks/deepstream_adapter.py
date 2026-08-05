"""DeepStream framework adapter -- written to the documented GStreamer +
`pyds` pipeline shape, NOT execution-verified in this environment, and the
most speculative adapter in this project.

DeepStream is a GStreamer-based video *streaming* analytics SDK, not a
simple load-model/call-predict API like every other framework here: a
real DeepStream app builds a GStreamer pipeline (`nvstreammux` ->
`nvinfer` -> ...) where `nvinfer` is configured via a `.txt` config file
pointing at a TensorRT engine (the same engine
frameworks/tensorrt_adapter.py already knows how to build) or an ONNX
model DeepStream compiles itself, and results arrive as buffer metadata
via `pyds.gst_buffer_get_nvds_batch_meta(...)` in a pad-probe callback,
not a return value. Adapting that to this project's synchronous
`predict(model, input_tensor) -> output_tensor` shape means pushing a
single frame through an `appsrc` and blocking on an `appsink` for its
metadata -- a legitimate but unusual way to use a pipeline designed for
continuous video, and the part of this adapter most likely to need
correction against a real DeepStream install.

Confirmed here: `pyds` isn't a PyPI package -- it ships inside the
DeepStream SDK container/install (Linux + NVIDIA GPU + DeepStream SDK,
typically via NVIDIA's DeepStream container image), none of which exist
in this environment. Treat this the same as
frameworks/executorch_adapter.py, but with lower confidence than any
other adapter in this project given the architectural gap above --
written to the documented pipeline shape, not verified, expect to correct
it against a real DeepStream install and current `pyds`/GStreamer Python
binding docs before relying on it.

Nothing DeepStream-specific is imported at module scope, so this module
still registers cleanly -- and shows up correctly in `python main.py
--list` -- without the SDK installed.
"""
import struct
import tempfile
from pathlib import Path

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}
_NVINFER_CONFIG_TEMPLATE = """\
[property]
gpu-id=0
onnx-file={onnx_path}
model-engine-file={engine_path}
batch-size=1
network-mode=0
num-detected-classes={num_classes}
"""


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class DeepStreamModel:
    def __init__(self, pipeline, appsrc, appsink, num_classes):
        self.pipeline = pipeline
        self.appsrc = appsrc
        self.appsink = appsink
        self.num_classes = num_classes


class DeepStreamAdapter(FrameworkAdapter):
    def __init__(self):
        import pyds  # noqa: F401 -- fail fast here if not installed
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._Gst = Gst

    def load_model(self, architecture_entry, config):
        import torch

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)
        num_classes = int(torch_model(dummy_input).shape[-1])

        work_dir = Path(tempfile.mkdtemp())
        onnx_path = work_dir / "model.onnx"
        config_path = work_dir / "nvinfer_config.txt"
        with torch.no_grad():
            torch.onnx.export(
                torch_model, dummy_input, str(onnx_path), input_names=["input"], output_names=["output"]
            )
        config_path.write_text(
            _NVINFER_CONFIG_TEMPLATE.format(
                onnx_path=onnx_path, engine_path=work_dir / "model.engine", num_classes=num_classes
            )
        )

        Gst = self._Gst
        pipeline = Gst.parse_launch(
            "appsrc name=src ! nvvideoconvert ! nvstreammux name=mux batch-size=1 "
            f"! nvinfer name=infer config-file-path={config_path} "
            "! nvvideoconvert ! appsink name=sink"
        )
        appsrc = pipeline.get_by_name("src")
        appsink = pipeline.get_by_name("sink")
        pipeline.set_state(Gst.State.PLAYING)

        return DeepStreamModel(pipeline, appsrc, appsink, num_classes)

    def predict(self, model, input_tensor):
        import numpy as np
        import pyds
        import torch
        from gi.repository import Gst

        array = input_tensor.detach().cpu().numpy().astype(np.uint8)
        if array.ndim == 3:
            array = array[None, ...]

        buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(array).tobytes())
        model.appsrc.emit("push-buffer", buf)

        sample = model.appsink.emit("pull-sample")
        out_buf = sample.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(out_buf))

        scores = np.zeros(model.num_classes, dtype=np.float32)
        frame_meta_list = batch_meta.frame_meta_list
        if frame_meta_list is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            classifier_meta_list = frame_meta.classifier_meta_list
            while classifier_meta_list is not None:
                classifier_meta = pyds.NvDsClassifierMeta.cast(classifier_meta_list.data)
                label_meta_list = classifier_meta.label_info_list
                while label_meta_list is not None:
                    label_meta = pyds.NvDsLabelInfo.cast(label_meta_list.data)
                    if label_meta.result_class_id < model.num_classes:
                        scores[label_meta.result_class_id] = label_meta.result_prob
                    label_meta_list = label_meta_list.next
                classifier_meta_list = classifier_meta_list.next

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


@FRAMEWORKS.register("DeepStream", implemented=True, organization="NVIDIA", platforms=["jetson"])
def build_deepstream_adapter(**kwargs):
    return DeepStreamAdapter()
