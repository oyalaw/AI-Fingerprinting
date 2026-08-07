"""SpeechBrain speech-framework adapter -- real implementation, wired into
execution via architectures/whisper.py's dispatch (same pattern as
graph_frameworks/pytorch_geometric_adapter.py's GCN dispatch and
llm_frameworks/fastchat_adapter.py's GPT-2 dispatch).

Real SpeechBrain layers (`speechbrain.nnet.CNN.Conv1d`,
`speechbrain.nnet.RNN.LSTM`, `speechbrain.nnet.linear.Linear`), NOT
SpeechBrain's higher-level `lobes.models.CRDNN`/pretrained ASR pipelines:
importing `CRDNN` was tried first and found to hard-fail here --
confirmed directly it's not this project's own bug: defining that class
triggers SpeechBrain's lazy-module machinery to eagerly resolve
`speechbrain.integrations.k2_fsa`, an unrelated optional decoding backend,
which needs the separate `k2` package. Installed `k2` to check: its own
PyPI wheel is tagged `py38` and imports a compiled `_k2` extension that
isn't present in the wheel at all on this platform -- the same class of
"real project, dead PyPI wheel" situation as graph_frameworks/dgl_adapter.py.
Confirmed the fix: importing `speechbrain.nnet.RNN`/`speechbrain.nnet.CNN`
directly (not through the `sb.*` lazy alias CRDNN uses) avoids the broken
resolution entirely -- these are real, independently loadable SpeechBrain
building blocks.

architectures/whisper.py's postprocess() (in applications/speech_recognition.py)
decodes the model's output as Whisper-vocabulary token ids via
`WhisperTokenizer`. This model has no actual relationship to Whisper's
vocabulary -- the final Linear layer's output size (51865) is set to match
Whisper's vocab size purely so the *shape* contract with the existing,
unmodified postprocess() holds, not because the two share any real
semantics. That's consistent with this project's existing policy: Whisper
itself is random-init and produces decoded gibberish for the same reason
(see applications/speech_recognition.py's docstring) -- this SpeechBrain
model is exactly as "meaningful" as the Whisper path it stands in for,
which is to say: real forward pass, real traffic, not real transcription.

Verified end-to-end with Whisper's actual input shape: (1, 3000, 80)
(transposed from the wire's (80, 3000) log-mel features) through a strided
Conv1d (3000 -> 375 timesteps) -> LSTM -> Linear(51865) -> argmax, first 10
timesteps taken as token ids (matching Whisper's own _MAX_NEW_TOKENS=10),
and confirmed those decode through the real WhisperTokenizer + this
project's actual ASCII-sanitization step without crashing.
"""
import torch

from core.registry import SPEECH_FRAMEWORKS

_VOCAB_SIZE = 51865
_OUTPUT_TOKENS = 10


class _SpeechBrainASRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from speechbrain.nnet.CNN import Conv1d
        from speechbrain.nnet.linear import Linear
        from speechbrain.nnet.RNN import LSTM

        example_shape = (1, 3000, 80)
        self.conv = Conv1d(out_channels=32, kernel_size=5, stride=8, input_shape=example_shape)

        conv_out_shape = (1, 375, 32)
        self.lstm = LSTM(hidden_size=32, input_shape=conv_out_shape)
        self.proj = Linear(n_neurons=_VOCAB_SIZE, input_shape=conv_out_shape)

    def forward(self, input_features):
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        # Wire format is (batch, 80 mel bins, 3000 frames); SpeechBrain's
        # layers expect (batch, time, channels).
        x = input_features.transpose(1, 2)
        x = self.conv(x)
        x, _ = self.lstm(x)
        logits = self.proj(x)
        return logits.argmax(dim=-1)[:, :_OUTPUT_TOKENS]


class SpeechBrainAdapter:
    def build_whisper(self):
        return _SpeechBrainASRModel()


@SPEECH_FRAMEWORKS.register("SpeechBrain", implemented=True, application="Speech tasks")
def build_speechbrain_adapter(**kwargs):
    return SpeechBrainAdapter()
