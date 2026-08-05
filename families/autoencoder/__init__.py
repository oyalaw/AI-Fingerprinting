"""Autoencoder family: shared [0,1] pixel-scaling helper. Deliberately NOT
families/cnn's ImageNet mean/std normalization -- architectures/autoencoder.py
uses a Sigmoid decoder output, so both the encoder's input and what the
decoder is reconstructing need to stay in [0,1] for the comparison to be
meaningful, not ImageNet-mean-centered like the classification pipeline.
"""
import torch

from core.registry import FAMILIES


def scale_to_unit_range(array_hwc_uint8):
    """(H, W, 3) uint8 -> (3, H, W) float32 tensor scaled to [0, 1]."""
    return torch.from_numpy(array_hwc_uint8.copy()).float().permute(2, 0, 1) / 255.0


FAMILIES.add("Autoencoder", implemented=True, description="Autoencoders")
