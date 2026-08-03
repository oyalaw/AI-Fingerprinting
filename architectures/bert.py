"""BERT -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("BERT is not yet implemented.")


ARCHITECTURES.register(
    "BERT", implemented=False, family="Transformer", framework="PyTorch"
)(build)
