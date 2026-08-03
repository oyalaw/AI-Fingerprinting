"""Ollama -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Ollama is not yet implemented.")


LLM_FRAMEWORKS.register("Ollama", implemented=False, use_case='Local LLM serving')(build)
