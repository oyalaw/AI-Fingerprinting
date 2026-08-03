"""FastChat -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("FastChat is not yet implemented.")


LLM_FRAMEWORKS.register("FastChat", implemented=False, use_case='LLM serving')(build)
