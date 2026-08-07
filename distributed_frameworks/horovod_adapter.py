"""Horovod -- deliberately left a stub after real investigation.

Checked directly: `pip install horovod` builds from source (no wheel for
this platform) and fails immediately with `RuntimeError: Failed to
install temporary CMake. Please update your CMake to 3.13+ or set
HOROVOD_CMAKE appropriately` -- no CMake on this machine. Same class of
blocker as apache-tvm/APPFL/FedML/YOLOX: a missing build toolchain, not a
hardware/OS gate. Installing CMake just to unblock this one package was
judged the same kind of invasive fix as those -- not done unprompted.
Horovod is also fundamentally MPI/NCCL-based multi-GPU tooling even once
built, so getting past this error would very likely just surface the next
Windows-incompatibility layer, the same pattern
distributed_frameworks/deepspeed_adapter.py and byteps_adapter.py already
found for this whole category of framework.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Horovod's build requires CMake, not installed -- see this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("Horovod", implemented=False, organization="Uber")(build)
