"""TensorRT framework adapter -- real implementation.

Builds a TensorRT engine by exporting the selected architecture's PyTorch
module to ONNX, then compiling it with TensorRT's builder API (this is the
standard TensorRT workflow: TensorRT doesn't define models itself, it
compiles them from an exchange format). Requires:

  - An NVIDIA GPU with the TensorRT Python package (`tensorrt`) available.
    On Jetson boards this ships as a system package as part of JetPack --
    build your venv with `--system-site-packages` (or don't use a venv)
    rather than `pip install tensorrt`, which isn't a plain PyPI wheel for
    Jetson's ARM64 + Jetson-specific CUDA/cuDNN build.
  - `pycuda` for host<->device memory transfers.
  - Targets the TensorRT 8.5+ / 10.x Python API (named-tensor addressing,
    `execute_async_v3`) that current JetPack 5.1+/6.x ships. A small
    compatibility shim covers the `create_network()` signature change
    between TensorRT 8.x (takes an EXPLICIT_BATCH flag) and 10.x (no
    flag needed, explicit batch is the only mode).

Every import of tensorrt/pycuda/torch is lazy (inside __init__/methods),
so this module still registers cleanly -- and shows up correctly in
`python main.py --list` -- on a machine with no GPU at all. Only actually
building or running a TensorRT model requires the real hardware/SDK.
"""
import io
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class TensorRTModel:
    """Wraps a built engine + execution context + I/O binding metadata."""

    def __init__(self, engine, context, input_name, output_name, input_shape):
        self.engine = engine
        self.context = context
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = input_shape


class TensorRTAdapter(FrameworkAdapter):
    def __init__(self):
        import tensorrt as trt

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)

    def _create_network(self, builder):
        trt = self._trt
        try:
            # TensorRT 8.x: explicit batch must be requested via a flag.
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            return builder.create_network(flag)
        except (AttributeError, TypeError):
            # TensorRT 10.x: explicit batch is the only mode, no flag argument.
            return builder.create_network()

    def load_model(self, architecture_entry):
        import torch

        torch_model = architecture_entry.build(self)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        dummy_input = torch.randn(1, *input_shape)

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            torch_model,
            dummy_input,
            onnx_buffer,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        onnx_bytes = onnx_buffer.getvalue()

        trt = self._trt
        builder = trt.Builder(self._logger)
        network = self._create_network(builder)
        parser = trt.OnnxParser(network, self._logger)
        if not parser.parse(onnx_bytes):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT failed to parse the ONNX export:\n{errors}")

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        input_tensor = network.get_input(0)
        full_shape = (1, *input_shape)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, full_shape, full_shape, full_shape)
        builder_config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, builder_config)
        if serialized_engine is None:
            raise RuntimeError(
                "TensorRT engine build failed. This requires an NVIDIA GPU with a "
                "working TensorRT installation (e.g. a Jetson board with JetPack)."
            )

        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        output_name = network.get_output(0).name

        return TensorRTModel(
            engine=engine,
            context=context,
            input_name=input_tensor.name,
            output_name=output_name,
            input_shape=full_shape,
        )

    def predict(self, model, input_tensor):
        import numpy as np
        import pycuda.autoinit  # noqa: F401 -- initializes the CUDA context for this thread
        import pycuda.driver as cuda
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        model.context.set_input_shape(model.input_name, array.shape)
        output_shape = tuple(model.context.get_tensor_shape(model.output_name))
        output = np.empty(output_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(array.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        try:
            model.context.set_tensor_address(model.input_name, int(d_input))
            model.context.set_tensor_address(model.output_name, int(d_output))

            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, array, stream)
            model.context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
        finally:
            d_input.free()
            d_output.free()

        return torch.from_numpy(output)

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


@FRAMEWORKS.register("TensorRT", implemented=True, organization="NVIDIA", platforms=["jetson"])
def build_tensorrt_adapter(**kwargs):
    return TensorRTAdapter()
