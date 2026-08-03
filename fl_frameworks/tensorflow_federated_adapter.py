"""TensorFlow Federated -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("TensorFlow Federated is not yet implemented.")


FL_FRAMEWORKS.register("TensorFlow Federated", implemented=False, organization='Google')(build)
