"""MLP family: no shared preprocessing helper needed -- architectures/mlp.py
reuses applications/activity_recognition.py's existing UCI HAR windowing
unchanged (same (timesteps, channels) tensor architectures/lstm.py and
architectures/gru.py already consume), just flattened before the first
Linear layer instead of fed through a recurrent cell."""
from core.registry import FAMILIES

FAMILIES.add("MLP", implemented=True, description="Multi-Layer Perceptrons")
