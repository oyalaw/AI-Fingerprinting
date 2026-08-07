"""ExLlama -- deliberately left a stub after real investigation. The
original ExLlama is deprecated in favor of ExLlamaV2 (the pip package is
`exllamav2`; there's no maintained `exllama` package to install), so this
was checked against the real, current project.

`pip install exllamav2` succeeds cleanly (pure Python wheel), but
`import exllamav2` fails: it JIT-compiles a C++ extension via
`torch.utils.cpp_extension` on first import, which needs `ninja` (not
installed) and, past that, a real C/C++ compiler toolchain (also not
installed, confirmed repeatedly elsewhere in this project). Even setting
the missing build tools aside, the import's own warning is explicit and
unconditional: "The installed version of PyTorch is 2.13.0+cpu and does
not support CUDA or ROCm" -- ExLlama's quantized-inference kernels are
CUDA/ROCm-only by design, so this would still need an NVIDIA/AMD GPU even
with a working compiler, the same hardware gate as
frameworks/tensorrt_adapter.py.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ExLlamaV2 needs a C++ compiler+ninja to JIT-build its extension, and its "
        "kernels are CUDA/ROCm-only regardless -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("ExLlama", implemented=False, use_case="Quantized LLMs")(build)
