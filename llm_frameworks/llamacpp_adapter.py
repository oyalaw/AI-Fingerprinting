"""llama.cpp -- deliberately left a stub after real investigation.

Checked directly: `pip install llama-cpp-python` (the real Python binding
package) falls back to a source build for this Python/platform (no
matching prebuilt wheel) and fails immediately with `CMake Error:
CMAKE_C_COMPILER not set` / `CMAKE_CXX_COMPILER not set` -- no C/C++
compiler on this machine, the same class of blocker as
distributed_frameworks/horovod_adapter.py and llm_frameworks/
exllama_adapter.py. Somewhat ironic given llama.cpp is specifically
designed to be the CPU-friendly option in this table (its whole premise
is quantized CPU inference, no GPU needed) -- the blocker here is purely
the missing build toolchain, not a GPU/OS gate like most of the rest of
this file.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "llama-cpp-python's source build needs a C/C++ compiler (CMake can't find "
        "one) -- see this module's docstring."
    )


LLM_FRAMEWORKS.register("llama.cpp", implemented=False, use_case="CPU inference")(build)
