"""Megatron LM -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Megatron LM is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("Megatron LM", implemented=False, organization='NVIDIA')(build)
