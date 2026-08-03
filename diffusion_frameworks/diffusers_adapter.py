"""Diffusers -- registered but not yet implemented."""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Diffusers is not yet implemented.")


DIFFUSION_FRAMEWORKS.register("Diffusers", implemented=False, application='HuggingFace')(build)
