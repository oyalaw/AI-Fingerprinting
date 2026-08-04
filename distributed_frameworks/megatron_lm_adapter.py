"""Megatron-LM -- registered but not yet implemented.

Checked directly: there is no `megatron-lm` (or similar) package on PyPI
at all -- `pip index versions megatron-lm` returns `No matching
distribution found`. NVIDIA distributes Megatron-LM as a GitHub
repository meant to be cloned and run in place
(github.com/NVIDIA/Megatron-LM), not a pip-installable library, and it's
built around multi-GPU/multi-node LLM pretraining (tensor + pipeline
parallelism) rather than the single-model-fits-on-one-device pattern this
project's other adapters use -- a real implementation would look quite
different in shape from ddp_adapter.py/fairscale_adapter.py, not just a
drop-in swap.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Megatron-LM is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("Megatron LM", implemented=False, organization="NVIDIA")(build)
