"""ComfyUI -- deliberately left a stub, not a gap: there's no pip package
to install at all, by design.

Checked directly: `pip install comfyui` returns "No matching distribution
found" -- not a version-constraint or build failure like
diffusion_frameworks/invokeai_adapter.py or graph_frameworks/dgl_adapter.py,
there's simply no PyPI distribution under this or any close variant of
the name. Consistent with ComfyUI's real nature: it's a git-clone-and-run
node-based GUI application (`git clone` + `python main.py` serving a web
UI), never designed or packaged as an importable library. Wiring this in
for real would mean running it as a subprocess/server and driving it
through its HTTP/websocket API, a fundamentally different integration
shape than every other framework in this registry (which are all
imported and called in-process) -- worth reconsidering as a role/
transport-level integration if this project ever needs that shape, not
as a stub-adapter fix.
"""
from core.registry import DIFFUSION_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ComfyUI has no importable pip package -- it's a standalone server "
        "application, not a library -- see this module's docstring."
    )


DIFFUSION_FRAMEWORKS.register(
    "ComfyUI", implemented=False, application="Workflow based generation"
)(build)
