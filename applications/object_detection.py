"""Object Detection application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class ObjectDetection(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Object Detection is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Object Detection is not yet implemented.")


APPLICATIONS.register("Object Detection", implemented=False)(ObjectDetection)
