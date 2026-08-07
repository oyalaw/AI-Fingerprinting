"""vLLM -- deliberately left a stub after real investigation, further
than most findings in this file got.

Checked directly: `pip install vllm` actually succeeds -- a large but
entirely pure-Python/prebuilt-wheel dependency tree (~80 packages),
surprising given vLLM's Linux/CUDA reputation, and `import vllm` works.
The real wall is deeper: `vllm.LLM` (the local-inference entry point)
transitively requires `uvloop`, and installing that directly gives an
unambiguous answer straight from uvloop's own setup.py: `RuntimeError:
uvloop does not support Windows at the moment`. Not a missing wheel, not
a build-toolchain gap, not a version ceiling -- a flat, permanent,
upstream "this library doesn't run on your OS" from a hard dependency
vLLM's async engine requires. No workaround available on Windows.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "vLLM requires uvloop, which explicitly does not support Windows -- "
        "see this module's docstring."
    )


LLM_FRAMEWORKS.register("vLLM", implemented=False, use_case="High throughput inference")(build)
