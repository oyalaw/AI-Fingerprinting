"""Stable Diffusion WebUI (AUTOMATIC1111) -- deliberately left a stub, not
a gap: there's no pip package to install at all, by design.

Checked directly: `pip install stable-diffusion-webui` (and `sdwebui`)
both return "No matching distribution found" -- same situation as
diffusion_frameworks/comfyui_adapter.py, not a version constraint or
build failure. Consistent with its real nature: a git-clone-and-run
Gradio web application (`git clone` + `webui.sh`/`webui.bat`), never
packaged as an importable library. It does expose an HTTP API once
running (`--api` flag), so the same "drive it as a subprocess/server"
integration shape noted in comfyui_adapter.py's docstring applies here
too, if this project ever wants that.
"""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Stable Diffusion WebUI has no importable pip package -- it's a standalone "
        "server application, not a library -- see this module's docstring."
    )


DIFFUSION_FRAMEWORKS.register(
    "Stable Diffusion WebUI", implemented=False, application="Image generation"
)(build)
