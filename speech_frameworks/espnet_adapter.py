"""ESPnet -- deliberately left a stub after real investigation.

Checked directly: `pip install espnet` fails building a transitive
dependency, `sentencepiece`, from source -- its build script invokes a
subprocess that isn't found (`FileNotFoundError: [WinError 2]`).
sentencepiece's real build process requires CMake to compile its C++
tokenizer core; this machine has none (confirmed repeatedly elsewhere in
this project -- e.g. distributed_frameworks/horovod_adapter.py,
llm_frameworks/exllama_adapter.py). Same root-cause class, one dependency
removed from ESPnet itself.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ESPnet's transitive sentencepiece dependency needs a build tool this "
        "machine doesn't have -- see this module's docstring."
    )


SPEECH_FRAMEWORKS.register("ESPnet", implemented=False, application="Speech recognition")(build)
