"""Kaldi -- registered but not yet implemented."""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("Kaldi is not yet implemented.")


SPEECH_FRAMEWORKS.register("Kaldi", implemented=False, application='ASR')(build)
