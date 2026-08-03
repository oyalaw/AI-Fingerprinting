"""ComfyUI -- registered but not yet implemented."""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ComfyUI is not yet implemented.")


DIFFUSION_FRAMEWORKS.register("ComfyUI", implemented=False, application='Workflow based generation')(build)
