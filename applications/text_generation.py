"""Text Generation application: real text prompt -> GPT-2 input tensors in,
continuation text out.

preprocess() reuses whatever real text the selected dataset provides (e.g.
IMDB review text) as a generation prompt, truncated/padded to a fixed
16-token window -- unlike Image Generation/Speech Recognition, a real seed
prompt is meaningful input here (there's something to condition generation
on), so this one doesn't synthesize its own input from scratch.

This project's wire protocol is one-tensor-in/one-tensor-out, but GPT-2
needs both input_ids and attention_mask -- stacked into a single
`(2, 16)` tensor here, unstacked by architectures/gpt2.py's
_GPT2Wrapper, the same two-tensor pattern already established for
DistilBERT.

postprocess() sanitizes the decoded text to ASCII before returning it,
proactively applying the exact lesson learned the hard way in
applications/speech_recognition.py: a random-init model's decoded output
is unconstrained token soup that can crash console logging with
UnicodeEncodeError on this Windows machine (cp1252).
"""
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_MAX_LENGTH = 16


class TextGeneration(Application):
    def __init__(self, max_length=_MAX_LENGTH):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def preprocess(self, raw_sample):
        prompt = raw_sample if isinstance(raw_sample, str) else str(raw_sample)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return torch.stack([encoded["input_ids"][0], encoded["attention_mask"][0]]).long()

    def postprocess(self, output_tensor):
        token_ids = output_tensor[0] if output_tensor.dim() == 2 else output_tensor
        text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return text.encode("ascii", errors="replace").decode("ascii")


APPLICATIONS.register("Text Generation", implemented=True)(TextGeneration)
