"""OpenVINO -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("OpenVINO is not yet implemented.")


FRAMEWORKS.register("OpenVINO", implemented=False, organization='Intel', platforms=['windows', 'linux', 'macos'])(build)
