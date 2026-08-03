"""BytePS -- registered but not yet implemented."""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("BytePS is not yet implemented.")


DISTRIBUTED_FRAMEWORKS.register("BytePS", implemented=False, organization='Bytedance')(build)
