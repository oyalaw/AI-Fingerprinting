"""Image Generation application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class ImageGeneration(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Image Generation is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Image Generation is not yet implemented.")


APPLICATIONS.register("Image Generation", implemented=False)(ImageGeneration)
