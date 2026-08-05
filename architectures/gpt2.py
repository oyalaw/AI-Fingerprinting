"""GPT-2 -- a new architecture entry beyond the original 14, same gap-fill
precedent as Autoencoder/YOLOv8-Seg/Whisper: Text Generation had no paired
architecture without it. Transformer family, decoder-only (unlike BERT/
DistilBERT's encoder-only classification heads).

Small random-init `GPT2Config` via `transformers` -- same policy as every
other architecture here, no pretrained MODEL weights, only the tokenizer's
fixed vocabulary downloaded (confirmed: ~3.3M params for this config vs
real gpt2's 124M).

Same autoregressive-generation-inside-forward() pattern as
architectures/whisper.py: `_GPT2Wrapper.forward()` runs the actual
`model.generate()` call internally (confirmed directly: a stacked
`(2, 16)` input_ids/attention_mask tensor with `max_new_tokens=10`
correctly produces a `(1, 26)` output -- 16 prompt tokens + 10 generated),
invisible to frameworks/pytorch_adapter.py, one tensor in, one tensor out.

This project's wire protocol is one-tensor-in/one-tensor-out, but GPT-2's
generate() needs both input_ids and attention_mask -- the same two-tensor
situation DistilBERT has (minus token_type_ids). applications/
text_generation.py stacks them into a single `(2, seq_len)` tensor;
_GPT2Wrapper unstacks it here, mirroring architectures/distilbert.py's
_DistilBertWrapper.
"""
import torch

from core.registry import ARCHITECTURES

_MAX_NEW_TOKENS = 10


class _GPT2Wrapper(torch.nn.Module):
    def __init__(self, gpt2_model, eos_token_id, max_new_tokens=_MAX_NEW_TOKENS):
        super().__init__()
        self.gpt2_model = gpt2_model
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens

    def forward(self, stacked_input):
        if stacked_input.dim() == 2:
            stacked_input = stacked_input.unsqueeze(0)
        input_ids = stacked_input[:, 0, :].long()
        attention_mask = stacked_input[:, 1, :].long()
        return self.gpt2_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.eos_token_id,
        )


def build(framework_adapter, config):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    gpt2_model = GPT2LMHeadModel(gpt2_config)
    return _GPT2Wrapper(gpt2_model, eos_token_id=tokenizer.eos_token_id)


ARCHITECTURES.register(
    "GPT-2", implemented=True, family="Transformer", framework="PyTorch", input_shape=(2, 16)
)(build)
