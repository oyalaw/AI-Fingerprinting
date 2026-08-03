"""Edge Impulse -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Edge Impulse is not yet implemented.")


FRAMEWORKS.register("Edge Impulse", implemented=False, organization='Edge Impulse', platforms=['linux', 'jetson'])(build)
