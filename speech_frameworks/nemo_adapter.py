"""NeMo -- deliberately left a stub, for a different reason than most
findings in this file: not a single clean blocker, but a long chain of
undeclared transitive dependencies that never bottomed out.

Checked directly, further than most: `pip install nemo-toolkit` succeeds,
and bare `import nemo` works. But `import nemo.collections.asr` (the
actual speech-recognition module, what a real adapter would need) failed
on a missing dependency four times in a row, each installed in turn to
see how deep it went: `wandb` -> `tensorboard` -> `hydra-core` ->
`lightning` -> `nv_one_logger` (NVIDIA's internal telemetry package, at
which point the chase was stopped). None of these are declared as
install-time dependencies of nemo-toolkit itself -- they're only
discovered by actually importing the submodule and reading each new
traceback. This is a different, messier category of finding than a clean
"needs a compiler" or "needs a GPU" wall: NeMo's own pip packaging
under-declares its real dependency footprint for the ASR collection
specifically, and there's no way to know how many more layers remain
short of continuing indefinitely.

Worth revisiting with more budget than this pass had, since nothing found
so far rules it out the way (for example) fl_frameworks/fedjax_adapter.py's
architectural mismatch does -- this is patience, not a wall.
"""
from core.registry import SPEECH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "nemo.collections.asr has a long chain of undeclared transitive "
        "dependencies (wandb, tensorboard, hydra, lightning, nv_one_logger, "
        "and counting) -- see this module's docstring."
    )


SPEECH_FRAMEWORKS.register("NeMo", implemented=False, application="NVIDIA speech")(build)
