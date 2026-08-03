"""Text Generation application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class TextGeneration(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Text Generation is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Text Generation is not yet implemented.")


APPLICATIONS.register("Text Generation", implemented=False)(TextGeneration)
