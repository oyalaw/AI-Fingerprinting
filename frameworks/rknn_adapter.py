"""RKNN -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("RKNN is not yet implemented.")


FRAMEWORKS.register("RKNN", implemented=False, organization='Rockchip', platforms=['linux'])(build)
