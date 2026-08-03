"""SGLang -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("SGLang is not yet implemented.")


LLM_FRAMEWORKS.register("SGLang", implemented=False, use_case='Structured generation')(build)
