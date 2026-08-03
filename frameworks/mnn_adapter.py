"""MNN -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("MNN is not yet implemented.")


FRAMEWORKS.register("MNN", implemented=False, organization='Alibaba', platforms=['android', 'ios', 'ipados'])(build)
