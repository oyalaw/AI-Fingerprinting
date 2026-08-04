"""Sentiment Analysis application: raw review text -> BERT input tensors in,
positive/negative label out.

This project's wire protocol is one-tensor-in, one-tensor-out (frameworks/
base.py's serialize/deserialize contract), but BERT needs three input
tensors (input_ids, token_type_ids, attention_mask). preprocess() stacks
them into a single (3, max_length) int64 tensor -- architectures/bert.py's
wrapper unstacks that back into the three named arguments the real model
needs before its forward pass.
"""
import torch

from applications.base import Application
from core.registry import APPLICATIONS

_LABELS = ("negative", "positive")


class SentimentAnalysis(Application):
    def __init__(self, max_length=32):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def preprocess(self, raw_sample):
        encoded = self.tokenizer(
            raw_sample,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return torch.stack(
            [encoded["input_ids"][0], encoded["token_type_ids"][0], encoded["attention_mask"][0]]
        ).long()

    def postprocess(self, output_tensor):
        probs = torch.softmax(output_tensor, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        return _LABELS[idx]


APPLICATIONS.register("Sentiment Analysis", implemented=True)(SentimentAnalysis)
