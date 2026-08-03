"""InvokeAI -- registered but not yet implemented."""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("InvokeAI is not yet implemented.")


DIFFUSION_FRAMEWORKS.register("InvokeAI", implemented=False, application='Image generation')(build)
