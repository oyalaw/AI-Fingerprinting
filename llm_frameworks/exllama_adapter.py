"""ExLlama -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ExLlama is not yet implemented.")


LLM_FRAMEWORKS.register("ExLlama", implemented=False, use_case='Quantized LLMs')(build)
