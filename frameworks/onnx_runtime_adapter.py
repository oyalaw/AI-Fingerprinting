"""ONNX Runtime -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ONNX Runtime is not yet implemented.")


FRAMEWORKS.register("ONNX Runtime", implemented=False, organization='Microsoft', platforms=['windows', 'linux', 'macos', 'jetson'])(build)
