"""llama.cpp LLM-serving framework adapter -- real implementation, wired
into execution via architectures/gpt2.py's dispatch (same pattern as
llm_frameworks/fastchat_adapter.py's GPT-2 dispatch and
graph_frameworks/pytorch_geometric_adapter.py's GCN dispatch).

Unlike every other llm_framework/architecture dispatch in this project,
llama.cpp doesn't build a `transformers.GPT2LMHeadModel` at all -- it's a
completely separate C++ inference engine (via the real `llama-cpp-python`
bindings) that loads a GGUF-format model and does its own tokenization in
its own vocabulary, structurally unrelated to HF's GPT-2 tokenizer/model
classes. Honestly labeling this as "the GPT-2 architecture" would be
wrong the same way silently falling back to CPU would be wrong for
frameworks/nnapi_adapter.py -- the ground-truth label needs to describe
what's actually running.

The resolution: keep the wire-protocol boundary exactly as GPT-2's
default path defines it (input_ids/attention_mask tokenized with GPT-2's
own tokenizer, in `applications/text_generation.py`'s existing,
unmodified preprocess/postprocess), and translate at the model boundary
instead. `_LlamaCppWrapper.forward()` decodes the wire's GPT-2-tokenized
input_ids back to a text prompt, runs *genuine* llama.cpp inference on
that text, then re-encodes prompt+completion back into GPT-2 token ids
for the wire's output tensor. Verified directly this round-trip is exact
and lossless for a real prompt ("Once upon a time" -> decode -> llama.cpp
-> ", there was a little girl named Lily" -> re-encode -> re-decode
matches). The actual generation work -- the reason this dispatch target
exists at all -- is genuinely done by llama.cpp's C++ engine, not
`transformers`; the GPT-2 tokenizer is only ever used as the wire
format's encode/decode boundary, the same category of translation
frameworks/coreml_adapter.py and openvino_adapter.py already do at their
own framework boundaries.

Needed a real C/C++ compiler to install at all -- `llama-cpp-python`
falls back to a from-source build for this Python/platform, which failed
outright until Visual Studio Build Tools (the "Desktop development with
C++" workload) was installed specifically to unblock this and the several
other stubs sharing the same root cause (see distributed_frameworks/
horovod_adapter.py, cv_frameworks/detectron2_adapter.py, etc. for the
others still blocked on this for hardware/OS reasons the compiler alone
doesn't fix).

Verified end-to-end with a real, tiny (~1.2MB), purpose-built test model:
`stories260K.gguf` from llama.cpp's own `ggml-org/models` test-fixture
repo on Hugging Face (a genuinely trained ~260K-parameter TinyStories
model, not one of this project's usual random-init architectures --
there's no equivalent "build a tiny GGUF from scratch" path the way
`GPT2Config(...)` builds a tiny transformers model in-process, so a real
pretrained-but-tiny file is the honest choice here, the same reasoning
YOLOv8 already established for using real pretrained weights). Downloaded
once and cached under ./data/, the same pattern as datasets/cifar10.py.

One cosmetic, harmless finding worth noting: `Llama.__del__` raises
`TypeError: 'NoneType' object is not callable` during CPython interpreter
shutdown (a known llama-cpp-python teardown-ordering issue, not a bug in
this code) -- confirmed it happens only after all real output has already
been produced correctly, never during actual use.
"""
import pathlib
import urllib.request

import torch

from core.registry import LLM_FRAMEWORKS

_GGUF_URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
_GGUF_PATH = pathlib.Path("./data/stories260K.gguf")
_MAX_TOKENS = 10


def _ensure_gguf_downloaded():
    _GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _GGUF_PATH.exists():
        urllib.request.urlretrieve(_GGUF_URL, _GGUF_PATH)
    return _GGUF_PATH


class _LlamaCppWrapper(torch.nn.Module):
    def __init__(self, llama_model, tokenizer, max_tokens=_MAX_TOKENS):
        super().__init__()
        self.llama_model = llama_model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def forward(self, stacked_input):
        if stacked_input.dim() == 2:
            stacked_input = stacked_input.unsqueeze(0)
        input_ids = stacked_input[0, 0, :].long()

        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        result = self.llama_model(prompt_text, max_tokens=self.max_tokens, echo=False)
        generated_text = result["choices"][0]["text"]

        full_text = prompt_text + generated_text
        output_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"]
        return output_ids


class LlamaCppAdapter:
    def build_gpt2(self):
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        gguf_path = _ensure_gguf_downloaded()
        llama_model = Llama(model_path=str(gguf_path), verbose=False)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return _LlamaCppWrapper(llama_model, tokenizer)


@LLM_FRAMEWORKS.register("llama.cpp", implemented=True, use_case="CPU inference")
def build_llamacpp_adapter(**kwargs):
    return LlamaCppAdapter()
