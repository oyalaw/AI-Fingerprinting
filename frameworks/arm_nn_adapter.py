"""Arm NN -- registered but not yet implemented."""
from core.registry import FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Arm NN is not yet implemented.")


FRAMEWORKS.register("Arm NN", implemented=False, organization='Arm', platforms=['android', 'linux', 'jetson'])(build)
