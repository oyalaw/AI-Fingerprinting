"""EasyFL -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("EasyFL is not yet implemented.")


FL_FRAMEWORKS.register("EasyFL", implemented=False, organization='Microsoft')(build)
