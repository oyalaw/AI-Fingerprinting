"""Text Generation Inference -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Text Generation Inference is not yet implemented.")


LLM_FRAMEWORKS.register("Text Generation Inference", implemented=False, use_case='HuggingFace')(build)
