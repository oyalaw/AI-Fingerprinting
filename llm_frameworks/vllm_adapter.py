"""vLLM -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("vLLM is not yet implemented.")


LLM_FRAMEWORKS.register("vLLM", implemented=False, use_case='High throughput inference')(build)
