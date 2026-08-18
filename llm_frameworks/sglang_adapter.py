"""SGLang -- deliberately left a stub after real investigation.

Originally blocked on Windows/Python 3.14: `pip install "sglang[all]"`
failed building a transitive dependency, `flashinfer_python`, which
itself required an exact pinned pre-release version of
`apache-tvm-ffi==0.1.0b15` with no matching distribution -- connected at
the time to the same finding on frameworks/tvm_adapter.py, Apache TVM's
own PyPI presence being unreliable on that platform/Python combination.

Re-checked on Ubuntu with Python 3.13: that specific blocker is gone --
`apache-tvm-ffi` now has real current releases (`pip index versions
apache-tvm-ffi` lists up through 0.1.13.post3), and `pip install
"sglang[all]" --dry-run` resolves a full dependency set cleanly with no
errors, choosing `apache-tvm-ffi-0.1.11`. But resolving it surfaces the
real wall underneath, the same one frameworks/tensorrt_adapter.py's GPU
requirement and llm_frameworks/vllm_adapter.py's plain-wheel CUDA target
already document elsewhere in this project: the resolved set is a CUDA
stack by default -- a CUDA-built `torch-2.11.0`/`torchvision`, a dozen-plus
`nvidia-cu*`/`nvidia-cuda-*` packages, `cuda-toolkit`, `flash-attn`,
`tilelang`, `sgl-deep-gemm` -- all built for an NVIDIA GPU this machine
doesn't have. Deliberately not run for real: it's several GB of
GPU-targeted packages that would install successfully but be unusable
here, the same "confirm the resolution, don't download gigabytes to prove
a foregone conclusion" judgment call `llm_frameworks/lmdeploy_adapter.py`
and `llm_frameworks/exllama_adapter.py`'s docstrings already make.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "SGLang's dependency resolution now succeeds on Ubuntu, but resolves a "
        "CUDA-targeted stack (torch, nvidia-cu*, flash-attn, ...) this machine has "
        "no GPU to run -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("SGLang", implemented=False, use_case="Structured generation")(build)
