"""llama.cpp -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("llama.cpp is not yet implemented.")


LLM_FRAMEWORKS.register("llama.cpp", implemented=False, use_case='CPU inference')(build)
