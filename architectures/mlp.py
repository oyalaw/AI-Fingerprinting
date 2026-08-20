"""MLP -- first and only MLP-family architecture. Reuses
applications/activity_recognition.py's UCI HAR pipeline unchanged, the
exact same (128, 9) timesteps-by-channels window architectures/lstm.py
and architectures/gru.py already consume -- flattened to a single
1152-dim vector and fed through plain Linear layers instead of a
recurrent cell, the simplest architecture in this project's whole zoo:
no convolution, no recurrence, no attention, just dense layers. Random
init, no training -- same policy as every other architecture here.
"""
import torch

from core.registry import ARCHITECTURES

_INPUT_SHAPE = (128, 9)


class _MLPClassifier(torch.nn.Module):
    def __init__(self, input_shape=_INPUT_SHAPE, hidden_size=128, num_classes=6):
        super().__init__()
        input_size = input_shape[0] * input_shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        return self.net(x.reshape(batch_size, -1))


def build(framework_adapter, config):
    return _MLPClassifier()


ARCHITECTURES.register(
    "MLP",
    implemented=True,
    family="MLP",
    framework="PyTorch",
    application="Activity Recognition",
    num_classes=6,
    input_shape=(128, 9),
)(build)
