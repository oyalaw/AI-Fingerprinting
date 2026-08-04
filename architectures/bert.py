"""BERT -- the second Transformer-family architecture, and the first NLP
one (paired with applications/sentiment_analysis.py + datasets/imdb.py).
Random init (`BertConfig` built from scratch, not `.from_pretrained()` for
the model itself) -- consistent with every other architecture here: no
pretrained MODEL weights, the point is realistic traffic, not language
understanding quality. Only the tokenizer's fixed vocabulary is downloaded
(a small, deterministic mapping, not a trained model -- the same category
of one-time download as CIFAR10's image files). Deliberately small
(hidden_size=64, 2 layers, 2 heads) for the same reason.

_BertWrapper unstacks the single (3, seq_len) tensor
applications/sentiment_analysis.py produces back into the three named
arguments (input_ids, token_type_ids, attention_mask)
BertForSequenceClassification actually needs, then returns just `.logits`
(not the full ModelOutput object) -- the same shape of fix as
architectures/yolov8.py's wrapper for YOLOv8's (detections, features)
output tuple, just on the input side instead of the output side.
"""
import torch

from core.registry import ARCHITECTURES


class _BertWrapper(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, stacked_input):
        if stacked_input.dim() == 2:
            stacked_input = stacked_input.unsqueeze(0)
        input_ids = stacked_input[:, 0, :].long()
        token_type_ids = stacked_input[:, 1, :].long()
        attention_mask = stacked_input[:, 2, :].long()
        output = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        return output.logits


def build(framework_adapter):
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
    )
    bert_model = BertForSequenceClassification(config)
    return _BertWrapper(bert_model)


ARCHITECTURES.register(
    "BERT", implemented=True, family="Transformer", framework="PyTorch", input_shape=(3, 32)
)(build)
