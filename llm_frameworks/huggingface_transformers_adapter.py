"""HuggingFace Transformers (llm_framework entry) -- real, but honestly
does NOT change behavior the way graph_frameworks/pytorch_geometric_adapter.py
does for GCN, the same nuance as speech_frameworks/whisper_adapter.py for
Whisper.

architectures/gpt2.py's default already builds a real
`transformers.GPT2LMHeadModel` directly -- that IS "the HuggingFace
Transformers framework." There's no second, distinct implementation for
this entry to dispatch to. Registered `implemented=True` because the
underlying model IS real and already verified (see architectures/gpt2.py),
not because selecting `llm_framework: HuggingFace Transformers` does
anything selecting nothing wouldn't already do.
"""
from core.registry import LLM_FRAMEWORKS


class HuggingFaceTransformersAdapter:
    """Not consulted by architectures/gpt2.py's build() -- there is no
    alternate implementation to dispatch to. See module docstring."""


@LLM_FRAMEWORKS.register("HuggingFace Transformers", implemented=True, use_case="NLP")
def build_huggingface_transformers_adapter(**kwargs):
    return HuggingFaceTransformersAdapter()
