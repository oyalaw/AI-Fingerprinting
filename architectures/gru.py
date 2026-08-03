"""GRU -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("GRU is not yet implemented.")


ARCHITECTURES.register(
    "GRU", implemented=False, family="RNN", framework="PyTorch"
)(build)
