"""DeepSpeed -- registered but not yet implemented.

Checked directly: `pip install deepspeed` fails at the build-requirements
step with `AssertionError: Unable to pre-compile ops without torch
installed. Please install torch before attempting to pre-compile ops.` --
even though torch *is* installed in this environment (used by every other
adapter here). DeepSpeed's setup.py needs torch visible inside pip's
isolated build environment specifically, which `--no-build-isolation`
would likely work around, but DeepSpeed is fundamentally a Linux/CUDA
datacenter-training package (custom CUDA op compilation, NCCL) -- getting
past this one error would very likely just surface the next
Windows-incompatibility layer. Not attempted further; see
distributed_frameworks/colossalai_adapter.py and byteps_adapter.py for
what those next layers typically look like for this whole category of
framework on Windows.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DeepSpeed is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("DeepSpeed", implemented=False, organization="Microsoft")(build)
