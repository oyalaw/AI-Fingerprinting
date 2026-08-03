"""PaddleFL -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("PaddleFL is not yet implemented.")


FL_FRAMEWORKS.register("PaddleFL", implemented=False, organization='Baidu')(build)
