"""GCN -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("GCN is not yet implemented.")


ARCHITECTURES.register(
    "GCN", implemented=False, family="GNN", framework="PyTorch"
)(build)
