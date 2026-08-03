"""TensorRT LLM -- registered but not yet implemented."""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorRT LLM is not yet implemented.")


LLM_FRAMEWORKS.register("TensorRT LLM", implemented=False, use_case='NVIDIA inference')(build)
