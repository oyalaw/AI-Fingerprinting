"""Sentiment Analysis application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class SentimentAnalysis(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("Sentiment Analysis is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("Sentiment Analysis is not yet implemented.")


APPLICATIONS.register("Sentiment Analysis", implemented=False)(SentimentAnalysis)
