"""LSTM -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("LSTM is not yet implemented.")


ARCHITECTURES.register(
    "LSTM", implemented=False, family="RNN", framework="PyTorch"
)(build)
