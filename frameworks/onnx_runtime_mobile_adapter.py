"""ONNX Runtime Mobile -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ONNX Runtime Mobile is not yet implemented.")


FRAMEWORKS.register("ONNX Runtime Mobile", implemented=False, organization='Microsoft', platforms=['android', 'ios', 'ipados'])(build)
