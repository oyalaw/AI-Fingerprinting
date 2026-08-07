"""NeMo -- deliberately left a stub. Chased further than most findings in
this file, and it does bottom out in a real wall -- just one five layers
deep.

Checked directly: `pip install nemo-toolkit` succeeds, and bare
`import nemo` works. But `import nemo.collections.asr` (the actual
speech-recognition module, what a real adapter would need) hits a chain
of undeclared transitive dependencies -- none of them listed as
nemo-toolkit's own install-time requirements, only discoverable by
actually importing the submodule and reading each new traceback:

    wandb -> tensorboard -> hydra-core -> lightning -> nv_one_logger

The first four all installed fine. The fifth, `nv_one_logger` (NVIDIA's
internal telemetry/logging package, imported unconditionally by
`nemo.lightning.one_logger_callback`, itself pulled in unconditionally by
`nemo.core.classes.modelPT` -- no lazy-loading escape hatch the way
speech_frameworks/speechbrain_adapter.py found for SpeechBrain's CRDNN
lobe), has zero PyPI distribution at all: `pip install nv-one-logger`
returns "No matching distribution found", consistent with it being an
internal-only or private-index NVIDIA package never published publicly.
That's a genuine, terminal wall, not a resolvable gap -- there's no
version of this chain that completes without a package that isn't public.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "nemo.collections.asr's import chain terminates at nv_one_logger, an "
        "NVIDIA-internal package with no public PyPI distribution -- see this "
        "module's docstring."
    )


SPEECH_FRAMEWORKS.register("NeMo", implemented=False, application="NVIDIA speech")(build)
