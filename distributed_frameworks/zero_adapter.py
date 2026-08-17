"""ZeRO -- registered but not yet implemented, and won't get its own entry.

ZeRO (Zero Redundancy Optimizer) is a technique from Microsoft's paper,
not a standalone installable package -- its reference implementation
ships as part of DeepSpeed (`deepspeed`'s ZeRO stage 1/2/3 configs).
Re-investigated on Ubuntu alongside deepspeed_adapter.py: DeepSpeed's
Windows blocker is now resolved and ZeRO stage 1 runs for real through
that adapter (`distributed_framework: DeepSpeed`, verified end-to-end via
`main.py` -- see that module's docstring). FairScale's `OSS` optimizer,
already implemented in distributed_frameworks/fairscale_adapter.py, is
itself an independent implementation of the same ZeRO stage 1 technique
(optimizer state sharding) -- so this technique is now covered twice over,
under two different real framework names, exactly as intended by keeping
them as separate registry entries per the project's original framework
table. `ZeRO` itself stays a stub since there's no standalone `zero`
package to point this entry's own `build()` at.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ZeRO has no standalone package -- see distributed_frameworks/deepspeed_adapter.py "
        "(ZeRO stage 1/2/3, implemented) and distributed_frameworks/fairscale_adapter.py "
        "(ZeRO stage 1, implemented) in this module's docstring."
    )


DISTRIBUTED_FRAMEWORKS.register("ZeRO", implemented=False, organization="Microsoft")(build)
