"""DeepSpeed -- registered but not yet implemented. The original
build-isolation blocker is resolved, and the predicted next
Windows-incompatibility layer is now confirmed directly rather than just
inferred.

`pip install --no-build-isolation deepspeed` (with a real C/C++ compiler
now installed -- Visual Studio Build Tools, added to unblock several
other stubs) gets past the original `AssertionError: Unable to pre-compile
ops without torch installed` cleanly. It then fails deeper, inside its own
`build_ext` step: `torch.utils.cpp_extension` logs `Error checking
compiler version for cl` / `FileNotFoundError: [WinError 2]` while probing
`cl`, then setuptools' MSVC linker itself throws `IndexError: list index
out of range` in `link_shared_object` -- a symptom of zero object files
having been produced to link, not a missing-compiler error (the same
compiler successfully built cv_frameworks/detectron2_adapter.py's ops
moments earlier in this same environment). Points at DeepSpeed's own
op-builder enumerating zero compilable sources for a Windows/no-CUDA
target rather than a real Linux-only failure it should otherwise
short-circuit around -- consistent with the original prediction that
DeepSpeed is fundamentally a Linux/CUDA datacenter-training package
(custom CUDA op compilation, NCCL) and getting past the first error would
surface exactly this kind of next Windows-incompatibility layer. Verified
the failed build left no side effects on the environment (torch/
torchvision/numpy versions unchanged, deepspeed itself not installed). See
distributed_frameworks/colossalai_adapter.py and byteps_adapter.py for
what this same category of framework's next layers typically look like on
Windows.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DeepSpeed is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("DeepSpeed", implemented=False, organization="Microsoft")(build)
