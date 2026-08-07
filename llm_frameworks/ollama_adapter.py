"""Ollama -- deliberately left a stub: the pip package is a remote client,
not the engine itself.

Checked directly: `pip install ollama` succeeds cleanly, and
`ollama.Client(host=None, **kwargs)`'s constructor confirms what the
package actually is -- an HTTP client (built on httpx) that talks to a
separately-running Ollama server (a Go binary distributed via Ollama's
own installer, not a pip package, normally listening on
localhost:11434). Same "remote client for external infrastructure this
project doesn't have" situation as fl_frameworks/fate_adapter.py, just
for a single-machine server instead of a cluster. There's nothing to
adapt in-process; genuinely using this would mean the user has Ollama's
real server running separately and this project's HTTP client just talks
to it -- a fundamentally different integration shape than every other
framework in this registry.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Ollama's pip package is an HTTP client for a separately-running Ollama "
        "server, not a local inference engine -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("Ollama", implemented=False, use_case="Local LLM serving")(build)
