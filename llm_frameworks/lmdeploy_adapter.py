"""LMDeploy -- deliberately left a stub after real investigation.

Checked directly: `pip install lmdeploy` fails dependency resolution --
it hard-requires `ray`, which has no wheel at all for this Python version
in this environment. Same root cause already documented on
distributed_frameworks/ray_train_adapter.py; nothing specific to
LMDeploy itself to add.

Re-checked on Ubuntu after Ray's wheel wall resolved (see
distributed_frameworks/ray_train_adapter.py's docstring): that specific
wall is gone, `pip install lmdeploy` gets past dependency resolution and
starts downloading real wheels -- but the resolved set (`lmdeploy-0.15.0`)
pulls a full CUDA-targeting stack by default: a GPU build of `torch`
(2.10.0, not the CPU wheel this project's own `torch` install uses),
`nvidia-cublas-cu12`/`nvidia-cudnn-cu12`/`nvidia-cusparselt-cu12` (each
several hundred MB to ~1GB), `cuda-bindings`, and `tilelang` (a CUDA
kernel-authoring DSL). Aborted before completing the download (multiple
GB, on a machine already tight on disk from this investigation pass) once
the pattern was unambiguous -- same class of wall as
llm_frameworks/vllm_adapter.py's finding: a real hardware gate (needs an
actual NVIDIA GPU), not an environment/wheel-availability one. Revisit on
a machine with an actual GPU, or if LMDeploy ever ships a genuine CPU-only
install path the way this project's other converter frameworks do.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "LMDeploy's Ray wall is resolved on Ubuntu, but it pulls a full CUDA-targeting "
        "stack (GPU torch, nvidia-cusparselt, tilelang) this machine has no GPU for -- "
        "see this module's docstring."
    )


LLM_FRAMEWORKS.register("LMDeploy", implemented=False, use_case="Efficient deployment")(build)
