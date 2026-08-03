"""Segmentation application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class Segmentation(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Segmentation is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Segmentation is not yet implemented.")


APPLICATIONS.register("Segmentation", implemented=False)(Segmentation)
