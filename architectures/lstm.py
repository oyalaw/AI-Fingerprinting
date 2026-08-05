"""LSTM -- first RNN-family architecture. Random init, small
(hidden_size=64, 1 layer) -- consistent with every other architecture
here, the point is realistic traffic, not classification accuracy.

Input: (batch, 128, 9), i.e. (batch, timesteps, channels) -- PyTorch's
batch_first=True convention. applications/activity_recognition.py already
produces (timesteps, channels) via families/rnn's shared transpose
helper, matching UCI HAR's 128-timestep/9-channel window shape (verified
directly against the real dataset -- see datasets/uci_har.py). Classifies
into UCI HAR's 6 activities.
"""
import torch

from core.registry import ARCHITECTURES


class _LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


def build(framework_adapter, config):
    return _LSTMClassifier()


ARCHITECTURES.register(
    "LSTM", implemented=True, family="RNN", framework="PyTorch", input_shape=(128, 9)
)(build)
