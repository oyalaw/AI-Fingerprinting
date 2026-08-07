"""Speech Recognition application: raw audio waveform -> Whisper log-mel
features in, transcribed text out.

preprocess() generates the "audio" itself rather than using the dataset
sample's actual content, the same design already established for
applications/image_generation.py (DDPM) -- the point is exercising
Whisper's real feature-extraction -> generate() -> decode pipeline with
correctly-shaped data, not achieving meaningful transcriptions from an
untrained model (same "traffic over accuracy" policy as every other
architecture here). `Synthetic`'s role is just driving the request count.

postprocess() sanitizes the decoded text to ASCII before returning it.
Confirmed directly this is a real, not hypothetical, concern: a random-init
Whisper's decoded output is unconstrained BPE token soup and crashed this
project's own logging with UnicodeEncodeError on this Windows console
(cp1252) during testing -- roles/client.py logs `predicted={result}`
straight to stdout, so an unsanitized result would non-deterministically
crash the client depending on what the untrained model happens to decode
to on a given run, not a cosmetic concern.

Registered `datasets=["Synthetic"]` for `--interactive`'s filtering --
narrower than what the code would technically tolerate (preprocess()
ignores `raw_sample` entirely, so it wouldn't crash on any dataset), but
`Synthetic` is the only one this project actually pairs it with anywhere
(experiment_matrix.yaml, README.md); offering e.g. `CIFAR10` here would
be a real, working combo that's also a nonsensical one to label "speech
recognition" traffic.
"""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_SAMPLE_RATE = 16000
_AUDIO_SECONDS = 1


class SpeechRecognition(Application):
    def __init__(self):
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        self.feature_extractor = WhisperFeatureExtractor(feature_size=80)
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def preprocess(self, raw_sample):
        raw_audio = np.random.default_rng().uniform(
            -1.0, 1.0, size=_SAMPLE_RATE * _AUDIO_SECONDS
        ).astype(np.float32)
        features = self.feature_extractor(raw_audio, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
        return features["input_features"][0]

    def postprocess(self, output_tensor):
        token_ids = output_tensor[0] if output_tensor.dim() == 2 else output_tensor
        text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return text.encode("ascii", errors="replace").decode("ascii")


APPLICATIONS.register(
    "Speech Recognition", implemented=True, datasets=["Synthetic"]
)(SpeechRecognition)
