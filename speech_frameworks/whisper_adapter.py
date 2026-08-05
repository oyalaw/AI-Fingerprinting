"""Whisper (speech_framework entry) -- real, but honestly does NOT change
behavior the way graph_frameworks/pytorch_geometric_adapter.py does for
GCN. That entry exists because PyTorch Geometric is a genuinely different
implementation from architectures/gcn.py's hand-rolled default -- there
were two real code paths to dispatch between.

Whisper has no such second path: architectures/whisper.py's default
*already* builds a real `transformers.WhisperForConditionalGeneration`
directly -- that IS "the Whisper framework." There's nothing distinct
for this entry to dispatch to; writing one anyway would mean either
duplicating architectures/whisper.py under a different name (dishonest
bookkeeping, the same reasoning frameworks/metal_performance_shaders_adapter.py
gave for staying a stub) or a no-op wrapper that changes nothing while
claiming to. Registered `implemented=True` because the underlying model
IS real and already verified (see architectures/whisper.py), not because
selecting `speech_framework: Whisper` does anything selecting nothing
wouldn't already do -- same honest nuance as cv_frameworks/
ultralytics_adapter.py for YOLOv8.
"""
from core.registry import SPEECH_FRAMEWORKS


class WhisperSpeechFrameworkAdapter:
    """Not consulted by architectures/whisper.py's build() -- there is no
    alternate implementation to dispatch to. See module docstring."""


@SPEECH_FRAMEWORKS.register("Whisper", implemented=True, organization="OpenAI")
def build_whisper_speech_framework_adapter(**kwargs):
    return WhisperSpeechFrameworkAdapter()
