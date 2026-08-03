"""NeMo -- registered but not yet implemented."""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("NeMo is not yet implemented.")


SPEECH_FRAMEWORKS.register("NeMo", implemented=False, application='NVIDIA speech')(build)
