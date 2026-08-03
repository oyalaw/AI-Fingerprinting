"""PySyft -- registered but not yet implemented."""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("PySyft is not yet implemented.")


FL_FRAMEWORKS.register("PySyft", implemented=False, organization='OpenMined')(build)
