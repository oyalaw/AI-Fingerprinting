"""vLLM -- deliberately left a stub after real investigation, further
than most findings in this file got. Re-investigated on Ubuntu (the
original findings below are all from this project's earlier Windows dev
machine): the specific Windows wall is now confirmed resolved, but a
different, deeper wall replaces it.

Original finding: `pip install vllm` succeeds and `import vllm` works, but
`vllm.LLM` (the local-inference entry point) transitively requires
`uvloop`, which gave an unambiguous `RuntimeError: uvloop does not support
Windows at the moment` straight from uvloop's own setup.py -- a flat,
permanent, upstream OS gate, not a missing wheel or version ceiling.

On Ubuntu, confirmed directly: `pip install uvloop` succeeds from a real
prebuilt `manylinux2014_x86_64` wheel (0.22.1) and `import uvloop` works
cleanly -- uvloop has never shipped a Windows wheel at all (confirmed via
PyPI's own file index: only macOS/manylinux/musllinux wheels exist for
every release), so this specific wall is fully gone on Linux, exactly as
predicted.

But `pip install vllm` itself reveals the real, deeper wall once actually
run here: PyPI's plain `vllm` wheel targets a CUDA GPU backend by default
and pulls a very large NVIDIA-specific dependency tree (`nvidia-cutlass-dsl`,
`nvidia-cuda-nvcc`, `nvidia-cuda-cccl`, `flashinfer-python`, `humming-kernels`,
`quack-kernels`, and more -- confirmed directly mid-resolution, several
hundred MB before even finishing) that's unusable on this GPU-less machine,
the same class of wall as `llm_frameworks/exllama_adapter.py`'s CUDA-only
kernels. vLLM's own PyPI summary does advertise x86/ARM/PowerPC CPU support,
but that path isn't the plain `pip install vllm` this project's other
adapters use -- it needs a from-source build with `VLLM_TARGET_DEVICE=cpu`
set, a materially larger undertaking than clearing the uvloop wall, and not
attempted here (consistent with this project's standing policy against
invasive, unprompted multi-step build chains -- see
fl_frameworks/pysyft_adapter.py's docstring for the same judgment call).
Revisit either on a machine with an actual NVIDIA/AMD/Intel GPU, or via a
deliberate, explicitly-scoped CPU-target source build.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "vLLM's uvloop dependency now works fine on Linux, but the plain PyPI "
        "wheel targets a CUDA GPU this machine doesn't have -- see this "
        "module's docstring."
    )


LLM_FRAMEWORKS.register("vLLM", implemented=False, use_case="High throughput inference")(build)
