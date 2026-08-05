"""Image Reconstruction application -- paired with architectures/autoencoder.py.
Also a new addition, not one of this project's original 8 applications:
none of them fit an autoencoder's natural task (encode an image, decode it
back out) any better than Node Classification fit GCN -- same justification
as datasets/karate_club.py's.

preprocess: raw HWC uint8 image -> [0,1]-scaled CHW tensor, via
families/autoencoder's scale_to_unit_range (not families/cnn's ImageNet
normalization -- the Sigmoid decoder output needs a [0,1] target to
compare against, not an ImageNet-mean-centered one).

postprocess: summarizes the reconstructed image the same way
applications/image_generation.py summarizes a generated one (shape/mean/
std) rather than returning the full tensor -- Application.postprocess only
ever sees the model's output, not the original input, so there's no
straightforward way to also report reconstruction error (which needs
both) without threading extra state through the pipeline; that's left as
a natural extension point, not something faked here.
"""
import numpy as np
import torch

from applications.base import Application
from core.registry import APPLICATIONS
from families.autoencoder import scale_to_unit_range


class ImageReconstruction(Application):
    def preprocess(self, raw_sample):
        if isinstance(raw_sample, np.ndarray):
            return scale_to_unit_range(raw_sample)
        return raw_sample  # already a tensor

    def postprocess(self, output_tensor):
        image = output_tensor.squeeze(0) if output_tensor.dim() == 4 else output_tensor
        return {
            "shape": tuple(image.shape),
            "mean_pixel": float(image.mean().item()),
            "std_pixel": float(image.std().item()),
        }


APPLICATIONS.register("Image Reconstruction", implemented=True)(ImageReconstruction)
