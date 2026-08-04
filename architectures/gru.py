"""GRU -- reuses the exact same UCI HAR/Activity Recognition pipeline as
architectures/lstm.py, swapping nn.LSTM for nn.GRU. GRU has no separate
cell state (just hidden state), one fewer thing forward() needs to
unpack from the RNN's return value -- the only real difference from the
paired LSTM implementation.
"""
import torch

from core.registry import ARCHITECTURES


class _GRUClassifier(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


def build(framework_adapter):
    return _GRUClassifier()


ARCHITECTURES.register(
    "GRU", implemented=True, family="RNN", framework="PyTorch", input_shape=(128, 9)
)(build)
