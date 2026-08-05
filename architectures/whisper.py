"""Whisper -- a new architecture entry beyond the original 13, the same
well-justified gap-fill precedent as Autoencoder and YOLOv8-Seg: the
taxonomy's Speech Recognition application has no paired architecture
without it. Transformer family (encoder-decoder), paired with
applications/speech_recognition.py.

Small random-init `WhisperConfig` (2 encoder layers, 2 decoder layers,
d_model=64) via `transformers` -- same policy as every other architecture
here, no pretrained MODEL weights, only the tokenizer's fixed vocabulary
downloaded (confirmed: ~3.6M params for this config vs whisper-tiny's real
39M, deliberately smaller/faster, consistent with BERT/DistilBERT/GCN all
being shrunk the same way).

Unlike DDPM's T-step denoising loop, Whisper's real inference (transcribing
audio) is *inherently* autoregressive generation, not a single forward
pass -- `_WhisperWrapper.forward()` runs the actual `model.generate()` call
internally (confirmed directly: `generate()` on a (1, 80, 3000) log-mel
input with `max_new_tokens=10` returns `(1, 10)` token ids), so this is a
real transcription call, not a truncated stand-in, and it's invisible to
frameworks/pytorch_adapter.py -- one tensor in (log-mel features, built by
applications/speech_recognition.py's preprocess), one tensor out (token
ids), same as every other architecture's wire contract.
"""
import torch

from core.registry import ARCHITECTURES

_MAX_NEW_TOKENS = 10


class _WhisperWrapper(torch.nn.Module):
    def __init__(self, whisper_model, max_new_tokens=_MAX_NEW_TOKENS):
        super().__init__()
        self.whisper_model = whisper_model
        self.max_new_tokens = max_new_tokens

    def forward(self, input_features):
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        return self.whisper_model.generate(input_features, max_new_tokens=self.max_new_tokens)


def build(framework_adapter, config):
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    whisper_config = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=2,
        encoder_attention_heads=2,
        decoder_layers=2,
        decoder_attention_heads=2,
        d_model=64,
        decoder_ffn_dim=128,
        encoder_ffn_dim=128,
        max_source_positions=1500,
        max_target_positions=448,
    )
    whisper_model = WhisperForConditionalGeneration(whisper_config)
    return _WhisperWrapper(whisper_model)


ARCHITECTURES.register(
    "Whisper", implemented=True, family="Transformer", framework="PyTorch", input_shape=(80, 3000)
)(build)
