"""Activity Recognition application: raw (channels, timesteps) sensor
window -> (timesteps, channels) tensor in, activity label out."""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS
from families.rnn import channels_first_to_timesteps_first

_ACTIVITY_LABELS = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)


class ActivityRecognition(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return channels_first_to_timesteps_first(raw_sample)
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        probs = torch.softmax(output_tensor, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        return _ACTIVITY_LABELS[idx]


APPLICATIONS.register("Activity Recognition", implemented=True)(ActivityRecognition)
