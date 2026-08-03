"""HuggingFace Transformers -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("HuggingFace Transformers is not yet implemented.")


LLM_FRAMEWORKS.register("HuggingFace Transformers", implemented=False, use_case='NLP')(build)
