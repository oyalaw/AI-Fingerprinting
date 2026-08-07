"""InvokeAI -- deliberately left a stub after real investigation.

Checked directly: `pip install InvokeAI` resolves against dozens of real
published versions (this is a genuinely PyPI-distributed, actively
maintained package, unlike ComfyUI/Stable Diffusion WebUI below), but
every single one pins a `Requires-Python` ceiling below this
environment's Python -- even the newest listed release
(6.14.0rc1) caps at `>=3.11,<3.13`. This project runs on a newer Python
than InvokeAI has added support for yet. Not a missing wheel, not a
build failure -- purely a Python-version ceiling. Revisit once InvokeAI
ships a release supporting this Python version (`pip index versions
invokeai` is the quick way to check), or if this project ever runs on an
older Python.
"""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "InvokeAI has no published release supporting this Python version -- "
        "see this module's docstring."
    )


DIFFUSION_FRAMEWORKS.register("InvokeAI", implemented=False, application="Image generation")(build)
