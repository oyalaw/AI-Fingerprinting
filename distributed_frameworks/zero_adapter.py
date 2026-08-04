"""ZeRO -- registered but not yet implemented.

ZeRO (Zero Redundancy Optimizer) is a technique from Microsoft's paper,
not a standalone installable package -- its reference implementation
ships as part of DeepSpeed (`deepspeed`'s ZeRO stage 1/2/3 configs), which
this project's distributed_frameworks/deepspeed_adapter.py already
tracks (blocked here on Windows -- see that module's docstring). FairScale's
`OSS` optimizer, already implemented in
distributed_frameworks/fairscale_adapter.py, is itself an independent
implementation of ZeRO stage 1 (optimizer state sharding) -- so the
technique this entry represents is partially covered there already, under
a different framework name, exactly as intended by keeping them as
separate registry entries per the project's original framework table.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ZeRO has no standalone package -- see distributed_frameworks/deepspeed_adapter.py "
        "(blocked) and distributed_frameworks/fairscale_adapter.py (ZeRO stage 1, implemented) "
        "in this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("ZeRO", implemented=False, organization="Microsoft")(build)
