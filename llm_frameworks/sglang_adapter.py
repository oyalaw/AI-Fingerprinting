"""SGLang -- deliberately left a stub after real investigation.

Checked directly: `pip install "sglang[all]"` fails building a transitive
dependency, `flashinfer_python`, which itself requires an exact pinned
pre-release version of `apache-tvm-ffi==0.1.0b15` with no matching
distribution. This connects directly to the already-documented finding on
frameworks/tvm_adapter.py: Apache TVM's own PyPI presence is unreliable
on this platform/Python combination. Same root cause, one dependency
removed.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "SGLang's flashinfer_python dependency needs an exact apache-tvm-ffi "
        "pre-release with no matching distribution -- see this module's docstring "
        "and frameworks/tvm_adapter.py's related finding."
    )


LLM_FRAMEWORKS.register("SGLang", implemented=False, use_case="Structured generation")(build)
