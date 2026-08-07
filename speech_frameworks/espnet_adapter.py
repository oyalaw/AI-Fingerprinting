"""ESPnet speech-framework adapter -- real implementation, wired into
execution via architectures/whisper.py's dispatch (same pattern as
speech_frameworks/speechbrain_adapter.py).

Previously a stub: `pip install espnet` used to fail building its
transitive `sentencepiece` dependency from source (needs CMake). After
installing a real C/C++ compiler (Visual Studio Build Tools, installed
specifically to unblock this and several other stubs sharing the "no
compiler" root cause), `pip install espnet` now succeeds outright and
`import espnet2` works cleanly (version 202511) -- confirmed directly,
including importing real building blocks:
`espnet2.asr.encoder.transformer_encoder.TransformerEncoder` and
`espnet2.asr.espnet_model.ESPnetASRModel`.

Uses `TransformerEncoder` directly (not the higher-level `ESPnetASRModel`/
task-recipe machinery, which expects a full training-config YAML and a
paired decoder+CTC setup well beyond what a single forward pass here
needs) -- the same "real building block, not the heavyweight pretrained
pipeline" choice speechbrain_adapter.py made for SpeechBrain. Confirmed
directly: `TransformerEncoder(input_size=80, ...)` run on a real
`(1, 3000, 80)` tensor returns a real `(1, T', output_size)` tensor (its
own `conv2d` input layer subsamples time ~4x, e.g. 3000 -> 749) plus real
output-length bookkeeping -- genuine ESPnet transformer encoder layers
(self-attention + positionwise feed-forward), not a hand-rolled stand-in.

Same wire-shape/postprocess situation as SpeechBrain: whisper.py's
postprocess() decodes output as Whisper-vocabulary token ids via
WhisperTokenizer, so the final Linear layer's output size (51865) is set
to match Whisper's vocab purely for shape compatibility, not because this
model shares any semantics with Whisper -- real forward pass, real
traffic, not real transcription (same policy as random-init Whisper
itself and the SpeechBrain path).
"""
import torch

from core.registry import SPEECH_FRAMEWORKS

_VOCAB_SIZE = 51865
_OUTPUT_TOKENS = 10


class _ESPnetASRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

        self.encoder = TransformerEncoder(
            input_size=80,
            output_size=32,
            attention_heads=2,
            linear_units=64,
            num_blocks=1,
            input_layer="conv2d",
        )
        self.proj = torch.nn.Linear(32, _VOCAB_SIZE)

    def forward(self, input_features):
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        # Wire format is (batch, 80 mel bins, 3000 frames); ESPnet's
        # TransformerEncoder expects (batch, time, feat).
        x = input_features.transpose(1, 2)
        ilens = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
        encoded, _, _ = self.encoder(x, ilens)
        logits = self.proj(encoded)
        return logits.argmax(dim=-1)[:, :_OUTPUT_TOKENS]


class ESPnetAdapter:
    def build_whisper(self):
        return _ESPnetASRModel()


@SPEECH_FRAMEWORKS.register("ESPnet", implemented=True, application="Speech recognition")
def build_espnet_adapter(**kwargs):
    return ESPnetAdapter()
