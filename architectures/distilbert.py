"""DistilBERT -- reuses the exact same Sentiment Analysis/IMDB pipeline as
architectures/bert.py, no changes needed to either. Same random-init
policy (small `DistilBertConfig` from scratch, only the tokenizer's fixed
vocabulary downloaded, not model weights).

The one real difference, confirmed directly: DistilBERT's forward() takes
only `input_ids` and `attention_mask` -- no `token_type_ids` at all (it
has no next-sentence-prediction pretraining objective, unlike BERT, so it
never needed segment embeddings). applications/sentiment_analysis.py still
produces the same (3, seq_len) stack every architecture in this pipeline
shares (kept generic rather than architecture-specific); _DistilBertWrapper
below just uses two of those three rows and ignores the token_type_ids one.
"""
import torch

from core.registry import ARCHITECTURES


class _DistilBertWrapper(torch.nn.Module):
    def __init__(self, distilbert_model):
        super().__init__()
        self.distilbert_model = distilbert_model

    def forward(self, stacked_input):
        if stacked_input.dim() == 2:
            stacked_input = stacked_input.unsqueeze(0)
        input_ids = stacked_input[:, 0, :].long()
        attention_mask = stacked_input[:, 2, :].long()
        output = self.distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits


def build(framework_adapter, config):
    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
        num_labels=2,
    )
    distilbert_model = DistilBertForSequenceClassification(model_config)
    return _DistilBertWrapper(distilbert_model)


ARCHITECTURES.register(
    "DistilBERT",
    implemented=True,
    family="Transformer",
    framework="PyTorch",
    application="Sentiment Analysis",
    input_shape=(3, 32),
)(build)
