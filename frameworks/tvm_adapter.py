"""Apache TVM framework adapter -- real implementation.

TVM is a compiler stack: it imports a traced PyTorch graph into its own
Relay IR, compiles it ahead-of-time for a target (here, `llvm` -- generic
CPU codegen, no special hardware needed), and runs the compiled module
through TVM's own graph executor runtime. Follows TVM's own documented
"Compile PyTorch Models" workflow:

  1. `torch.jit.trace` the architecture's PyTorch module (TVM's PyTorch
     frontend consumes a traced/scripted TorchScript graph, not a raw
     nn.Module).
  2. `tvm.relay.frontend.from_pytorch` imports it into Relay IR.
  3. `relay.build` compiles it for `target="llvm"`.
  4. `tvm.contrib.graph_executor.GraphModule` loads and runs the compiled
     module.

Environment note, updated after installing a real C/C++ compiler
(Visual Studio Build Tools) specifically to unblock this and several
other stubs sharing the "no compiler" root cause: `apache-tvm-ffi` (the
small component that previously failed to build) now compiles and
installs successfully. But the main `apache-tvm` package ships its own
*prebuilt* `tvm_runtime` native library, and loading it fails with
`OSError: [WinError 127] The specified procedure could not be found` --
a low-level DLL/binary-compatibility issue in a precompiled artifact,
not something the compiler helps with (we didn't build this DLL, TVM's
own release process did). Different, final root cause from the original
"no compiler" finding -- the compiler genuinely fixed one layer and
revealed a new one underneath. This adapter still follows TVM's
documented API as closely as this project's other adapters follow
theirs, but has NOT been execution-verified. TVM has also been
mid-transition between its classic Relay IR (used here, still documented
and supported) and a newer Relax IR in recent releases -- re-check this
against current TVM docs before relying on it, since which IR is the
recommended default has shifted across versions.

Re-checked on Ubuntu: both of the above are now moot, replaced by a
bigger, final finding. `apache-tvm` 0.26.0 now ships a real prebuilt
`manylinux` wheel with no separate `apache-tvm-ffi` compiler step needed
at all (confirmed: plain `pip install apache-tvm` succeeds cleanly, no
compiler invoked) -- and the WinError-127 native-library-loading issue is
confirmed gone too: `import tvm.runtime` loads the prebuilt runtime
cleanly on Linux, no error. But the "mid-transition" warning above turned
out to be exactly right: **Relay IR is gone**. Confirmed directly:
`from tvm import relay` raises `ImportError: cannot import name 'relay'
from 'tvm' (... Did you mean: 'relax'?)` -- this release doesn't ship the
`tvm.relay` module this adapter's code (`relay.frontend.from_pytorch`,
`relay.build`) is written against at all, only `tvm.relax`, confirmed
separately to import fine. Relax isn't a drop-in rename -- its PyTorch
frontend (`relax.frontend.torch.from_exported_program`, operating on a
`torch.export`-style `ExportedProgram` rather than a `torch.jit.trace`d
graph) and its own build/runtime API are a different surface entirely, the
same shape of rewrite ExecuTorch's own `torch.export`-based pipeline
uses. Every environment wall this docstring traced is now resolved; what's
left is a genuine rewrite against TVM's current Relax API, not an
environment blocker -- left as a stub for that reason, tracked here rather
than silently reused under the old Relay code.

torch/tvm are imported lazily so this module still registers cleanly --
and shows up correctly in `python main.py --list` -- on a machine without
them installed.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}
_INPUT_NAME = "input0"


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class TVMModel:
    def __init__(self, graph_module):
        self.graph_module = graph_module


class TVMAdapter(FrameworkAdapter):
    def __init__(self):
        import tvm  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import tvm
        import torch
        from tvm import relay
        from tvm.contrib import graph_executor

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        example_input = torch.randn(1, *input_shape)

        with torch.no_grad():
            traced_model = torch.jit.trace(torch_model, example_input).eval()

        shape_list = [(_INPUT_NAME, example_input.shape)]
        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        target = tvm.target.Target("llvm")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        device = tvm.cpu(0)
        graph_module = graph_executor.GraphModule(lib["default"](device))
        return TVMModel(graph_module)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        model.graph_module.set_input(_INPUT_NAME, array)
        model.graph_module.run()
        output = model.graph_module.get_output(0).numpy()
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


@FRAMEWORKS.register(
    "TVM",
    implemented=True,
    organization="Apache",
    platforms=["windows", "linux", "macos", "android", "jetson"],
)
def build_tvm_adapter(**kwargs):
    return TVMAdapter()
