"""Activity Recognition application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class ActivityRecognition(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Activity Recognition is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Activity Recognition is not yet implemented.")


APPLICATIONS.register("Activity Recognition", implemented=False)(ActivityRecognition)
