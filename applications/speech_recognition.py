"""Speech Recognition application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class SpeechRecognition(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Speech Recognition is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Speech Recognition is not yet implemented.")


APPLICATIONS.register("Speech Recognition", implemented=False)(SpeechRecognition)
