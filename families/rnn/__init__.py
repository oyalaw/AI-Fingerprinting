"""RNN family: shared metadata + sensor-window preprocessing reused by
every RNN architecture (LSTM, GRU, ...). UCI HAR's raw (channels,
timesteps) layout matches how families/cnn represents images
(channels-first); RNN architectures want (timesteps, channels) instead
(PyTorch's batch_first=True convention), so that transpose belongs here,
not duplicated per architecture.
"""
import torch

from core.registry import FAMILIES


def channels_first_to_timesteps_first(window):
    """(channels, timesteps) float32 array -> (timesteps, channels) tensor."""
    return torch.from_numpy(window.copy()).float().permute(1, 0)


FAMILIES.add("RNN", implemented=True, description="Recurrent Neural Networks")
