"""ESPnet -- registered but not yet implemented."""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ESPnet is not yet implemented.")


SPEECH_FRAMEWORKS.register("ESPnet", implemented=False, application='Speech recognition')(build)
