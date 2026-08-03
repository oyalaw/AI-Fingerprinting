"""ExecuTorch -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ExecuTorch is not yet implemented.")


FRAMEWORKS.register("ExecuTorch", implemented=False, organization='Meta', platforms=['android', 'ios', 'ipados', 'macos'])(build)
