"""TensorRT LLM -- deliberately left a stub. Same NVIDIA-GPU hardware
gate as frameworks/tensorrt_adapter.py, confirmed even more directly this
time: `pip install tensorrt-llm`'s own installer (it's a placeholder
package that fetches the real wheel from pypi.nvidia.com) self-diagnoses
and prints the exact reason before failing:

    nvidia-smi command not found. Ensure NVIDIA drivers are installed.

No NVIDIA GPU/drivers on this machine -- nothing to work around, real
hardware requirement stated by the package itself.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "TensorRT LLM needs an NVIDIA GPU (its own installer confirms: "
        "nvidia-smi not found) -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("TensorRT LLM", implemented=False, use_case="NVIDIA inference")(build)
