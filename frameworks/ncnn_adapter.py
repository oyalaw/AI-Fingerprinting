"""NCNN -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("NCNN is not yet implemented.")


FRAMEWORKS.register("NCNN", implemented=False, organization='Tencent', platforms=['android', 'ios', 'ipados', 'linux'])(build)
