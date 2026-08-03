"""Stable Diffusion WebUI -- registered but not yet implemented."""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Stable Diffusion WebUI is not yet implemented.")


DIFFUSION_FRAMEWORKS.register("Stable Diffusion WebUI", implemented=False, application='Image generation')(build)
