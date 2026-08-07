"""Coqui STT -- deliberately left a stub, not a gap: never published as a
pip package under any name variant tried.

Checked directly: `pip install stt` and `pip install coqui-stt` both
return "No matching distribution found". Consistent with Coqui AI (the
company behind this project, a continuation of Mozilla DeepSpeech) having
shut down in early 2024 and its packages having stopped receiving new
releases; no wheel exists for a Python version this new.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Coqui STT has no pip package for this Python version (Coqui AI shut down "
        "in 2024) -- see this module's docstring."
    )


SPEECH_FRAMEWORKS.register("Coqui STT", implemented=False, application="Speech to text")(build)
