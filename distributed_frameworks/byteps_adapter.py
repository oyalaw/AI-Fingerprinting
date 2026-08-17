"""BytePS -- registered but not yet implemented. Re-investigated on Ubuntu
(the original finding below was from this project's earlier Windows dev
machine): the `make` wall is resolved as predicted, and two further real
layers were found and confirmed underneath it.

Original finding: `pip install byteps` failed building its bundled
`ps-lite` C++ library, whose Makefile-based build invokes `make` --
`'make' is not recognized as an internal or external command` on Windows.

On Ubuntu, confirmed directly: `make` is present (part of `build-essential`),
and the `ps-lite`/`byteps` C++ sources now genuinely compile and link --
`server.cc`, `common.cc`, `cpu_reducer.cc`, etc. all build real `.o` files
and link into `c_lib.cpython-313-x86_64-linux-gnu.so` successfully. That
finding is fully resolved. Retrying then hit build-isolation blindness
(pip's isolated build sandbox can't see this project's already-installed
`torch`, the same class of gap documented in
cv_frameworks/yolox_adapter.py's docstring) -- `--no-build-isolation`
gets past that too, and BytePS's setup.py correctly detects the real
PyTorch install this time.

That reveals the actual final wall: BytePS 0.2.5.post15 (last released
~2022) calls `torch.utils.cpp_extension.include_paths(cuda=...)` in its own
`is_torch_cuda()` helper -- confirmed directly, current PyTorch's
`include_paths()` signature is `(device_type: str = "cpu",
torch_include_dirs=True)`, with no `cuda` keyword argument at all anymore
(renamed/restructured upstream since BytePS was last released):
`TypeError: include_paths() got an unexpected keyword argument 'cuda'`.
A genuine "old package's code no longer matches a current PyTorch API"
bug in BytePS's own source, same class of wall as
fl_frameworks/fedml_adapter.py's `pathtools`/`imp` finding -- not a
platform gate, and not something a compiler or pip flag can route around.
Confirmed no side effects from the attempt (byteps itself not left
installed). Revisit if BytePS ever publishes a release updated for a
current PyTorch, or if pinning an old `torch` compatible with BytePS's
assumptions becomes acceptable for this project (it wouldn't be --
every other adapter here depends on the current pin).
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "BytePS builds fine on Linux now, but its own is_torch_cuda() helper calls "
        "a torch.utils.cpp_extension.include_paths() keyword argument current PyTorch "
        "no longer has -- see this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("BytePS", implemented=False, organization="Bytedance")(build)
