"""ExLlama -- deliberately left a stub, now confirmed rather than
predicted to be a pure hardware gate. The original ExLlama is deprecated
in favor of ExLlamaV2 (the pip package is `exllamav2`; there's no
maintained `exllama` package to install), so this was checked against
the real, current project.

`pip install exllamav2` succeeds cleanly (pure Python wheel). Retried
`import exllamav2` after both a real C/C++ compiler (Visual Studio Build
Tools) and `ninja` (a normal pip package shipping a real prebuilt
`ninja.exe`, confirmed directly: `ninja --version` reports 1.13.0)
became available in this environment -- both build-tool blockers this
docstring originally cited are resolved. The JIT build now genuinely
proceeds (`torch.utils.cpp_extension`'s `_write_ninja_file_and_build_library`
runs for real) and fails at a different, later step:
`OSError: CUDA_HOME environment variable is not set` -- confirmed
directly, not just inferred from the accompanying warning this docstring
previously only quoted ("The installed version of PyTorch is 2.13.0+cpu
and does not support CUDA or ROCm"). ExLlamaV2's C++ extension
unconditionally references a CUDA toolkit path with no CPU-only code
path to fall back to -- its quantized-inference kernels are CUDA/ROCm-only
by design, the same hardware gate as frameworks/tensorrt_adapter.py.
Uninstalled again afterward (confirmed no side effects: torch/torchvision/
numpy versions unchanged). Revisit only on a machine with an actual
NVIDIA/AMD GPU and matching CUDA/ROCm toolkit -- no further build-tool
investigation is useful here.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "ExLlamaV2 builds fine now (compiler + ninja both available) but its "
        "kernels are CUDA/ROCm-only by design -- confirmed directly via "
        "CUDA_HOME OSError, see this module's docstring."
    )


LLM_FRAMEWORKS.register("ExLlama", implemented=False, use_case="Quantized LLMs")(build)
