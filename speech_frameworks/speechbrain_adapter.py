"""SpeechBrain -- registered but not yet implemented."""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("SpeechBrain is not yet implemented.")


SPEECH_FRAMEWORKS.register("SpeechBrain", implemented=False, application='Speech tasks')(build)
