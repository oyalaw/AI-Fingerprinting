"""Image Generation application: paired with architectures/ddpm.py.

preprocess ignores the dataset sample's actual pixel content and generates
fresh Gaussian noise instead -- the "input" to a generation request
genuinely is noise, not data. Synthetic's random images just drive the
request count/loop here, the same role Karate Club's repeated graph plays
for the transductive Node Classification application.

postprocess summarizes the generated image rather than returning the full
tensor, consistent with every other application here returning a compact
result rather than raw model output.
"""
import torch

from applications.base import Application
from core.registry import APPLICATIONS


class ImageGeneration(Application):
    def preprocess(self, raw_sample):
        return torch.randn(3, 32, 32)

    def postprocess(self, output_tensor):
        image = output_tensor.squeeze(0) if output_tensor.dim() == 4 else output_tensor
        return {
            "shape": tuple(image.shape),
            "mean_pixel": float(image.mean().item()),
            "std_pixel": float(image.std().item()),
        }


APPLICATIONS.register("Image Generation", implemented=True)(ImageGeneration)
