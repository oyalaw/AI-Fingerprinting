"""Whisper -- registered but not yet implemented."""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Whisper is not yet implemented.")


SPEECH_FRAMEWORKS.register("Whisper", implemented=False, application='Speech recognition')(build)
