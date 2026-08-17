"""Apache TVM framework adapter -- rewritten against current TVM (0.26.0),
real implementation, verified end-to-end.

TVM is a compiler stack: it imports a captured PyTorch graph into its own
IR, compiles it ahead-of-time for a target (here, `llvm` -- generic CPU
codegen, no special hardware needed), and runs the compiled module
through TVM's own runtime. This adapter originally followed TVM's classic
Relay IR workflow (`torch.jit.trace` -> `relay.frontend.from_pytorch` ->
`relay.build` -> `tvm.contrib.graph_executor`) -- see this project's own
git history for that version and the docstrings on
distributed_frameworks/colossalai_adapter.py and friends for how the
Windows->Ubuntu re-investigation pass found that Relay IR has been fully
removed from TVM 0.26.0 (`from tvm import relay` now raises
`ImportError: cannot import name 'relay' from 'tvm' ... Did you mean:
'relax'?`). Rewritten here against TVM's current Relax IR instead, TVM's
own documented replacement:

  1. `torch.export.export` the architecture's PyTorch module to an
     `ExportedProgram` (TVM's current PyTorch frontend consumes this,
     not a `torch.jit.trace`d graph -- the same ahead-of-time capture
     step frameworks/executorch_adapter.py uses).
  2. `tvm.relax.frontend.torch.from_exported_program(...,
     keep_params_as_input=True, unwrap_unit_return_tuple=True)` imports
     it into a Relax `IRModule`. `keep_params_as_input` keeps weights as
     explicit function arguments rather than baked-in constants, so
     `relax.frontend.detach_params` can pull them out as a plain
     parameter list to pass at call time (TVM's own documented pattern
     for this, confirmed directly against its docstring/example);
     `unwrap_unit_return_tuple` makes a single-output model return a
     plain tensor instead of a one-element `R.Tuple`, matching every
     other framework adapter's plain-tensor return.
  3. `tvm.compile(mod, target)` compiles the module -- confirmed directly
     this is the current top-level entry point (`relax.build` still
     exists too and does the same thing, `tvm.compile` is what current
     TVM's own API surface leads with).
  4. `tvm.relax.VirtualMachine(executable, device)` loads the compiled
     executable; calling `vm["main"](input_tensor, *param_tensors)` runs
     it, the Relax runtime's documented calling convention.

`tvm.nd.array` (the old Relay-era way to wrap a numpy array for TVM) is
also gone -- confirmed directly (`AttributeError: module 'tvm' has no
attribute 'nd'`); its replacement is `tvm.runtime.tensor(array, device)`.

**Verified end-to-end**: confirmed directly with a small hand-written
`nn.Linear` module first (compiled output matched the real PyTorch
forward pass exactly, `np.allclose(..., atol=1e-4)` true), then with this
project's real ResNet18 (via `PyTorch` framework's `load_model`,
`Image Classification`/`Synthetic`): the compiled Relax module's
`(1, 10)` output matches the real PyTorch model's own forward pass output
within `atol=1e-3` (real conv/batchnorm/pooling ops through the full
pipeline, not just a toy graph). Also confirmed a real client/server
roundtrip through `python main.py` itself (`framework: TVM`,
`architecture: ResNet18`, `dataset: Synthetic`) -- both requests served
correctly, `ground_truth.json` correctly recorded `"level1_framework":
"TVM"`. Note for anyone re-running this: `tvm.compile`'s LLVM codegen for
a ResNet18-sized graph takes real wall-clock time (single-digit seconds
once warm, longer on a cold run) -- give the server a real head start
before starting the client, the same as any other framework here.

torch/tvm are imported lazily so this module still registers cleanly --
and shows up correctly in `python main.py --list` -- on a machine without
them installed.
"""
import struct

from core.registry import FRAMEWORKS
from frameworks.base import FrameworkAdapter

_DTYPE_NAME_TO_CODE = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def _np_dtype_map():
    import numpy as np

    return {0: np.float32, 1: np.float64, 2: np.int64, 3: np.int32}


class TVMModel:
    def __init__(self, vm, param_tensors, device):
        self.vm = vm
        self.param_tensors = param_tensors
        self.device = device


class TVMAdapter(FrameworkAdapter):
    def __init__(self):
        import tvm  # noqa: F401 -- fail fast here if not installed

    def load_model(self, architecture_entry, config):
        import numpy as np
        import torch
        import tvm
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program

        torch_model = architecture_entry.build(self, config)
        torch_model.eval()

        input_shape = architecture_entry.meta.get("input_shape", (3, 224, 224))
        example_input = torch.randn(1, *input_shape)

        with torch.no_grad():
            exported_program = torch.export.export(torch_model, (example_input,))

        mod = from_exported_program(
            exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
        )
        mod, params = relax.frontend.detach_params(mod)

        target = tvm.target.Target("llvm")
        executable = tvm.compile(mod, target)

        device = tvm.cpu(0)
        vm = relax.VirtualMachine(executable, device)
        param_tensors = [tvm.runtime.tensor(p.numpy().astype(np.float32), device) for p in params["main"]]

        return TVMModel(vm, param_tensors, device)

    def predict(self, model, input_tensor):
        import numpy as np
        import torch
        import tvm

        array = input_tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3:
            array = array[None, ...]
        array = np.ascontiguousarray(array)

        tvm_input = tvm.runtime.tensor(array, model.device)
        output = model.vm["main"](tvm_input, *model.param_tensors)
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
    "TVM",
    implemented=True,
    organization="Apache",
    platforms=["windows", "linux", "macos", "android", "jetson"],
)
def build_tvm_adapter(**kwargs):
    return TVMAdapter()
