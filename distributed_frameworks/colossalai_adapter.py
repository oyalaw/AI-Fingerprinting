"""ColossalAI -- registered but not yet implemented.

Checked directly: `pip install colossalai` fails at the build-requirements
step with an explicit, unambiguous error from the package's own setup.py:
`RuntimeError: Windows is not supported yet. Please try again within the
Windows Subsystem for Linux (WSL).` This isn't a missing-wheel gap --
ColossalAI's maintainers deliberately gate native Windows out. Would need
WSL (or a Linux machine/container) to proceed at all.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ColossalAI is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("ColossalAI", implemented=False, organization="HPC AI Tech")(build)
