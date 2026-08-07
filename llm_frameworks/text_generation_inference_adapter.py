"""Text Generation Inference (TGI) -- deliberately left a stub, not a
gap: there's no pip package to install at all, by design.

Checked directly: `pip install text-generation-inference` returns "No
matching distribution found". Consistent with TGI's real nature: it's a
Rust/Python server distributed as a Docker image
(`ghcr.io/huggingface/text-generation-inference`), never packaged as an
importable Python library -- the same "standalone server application, not
a library" situation as diffusion_frameworks/comfyui_adapter.py and
stable_diffusion_webui_adapter.py.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Text Generation Inference has no pip package -- it's a Docker-distributed "
        "server, not a library -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("Text Generation Inference", implemented=False, use_case="HuggingFace")(build)
