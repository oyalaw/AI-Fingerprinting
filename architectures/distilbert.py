"""DistilBERT -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("DistilBERT is not yet implemented.")


ARCHITECTURES.register(
    "DistilBERT", implemented=False, family="Transformer", framework="PyTorch"
)(build)
