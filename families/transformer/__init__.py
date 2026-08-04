"""Transformer family: shared metadata for self-attention-based
architectures (ViT, BERT, DistilBERT, ...)."""
from core.registry import FAMILIES

FAMILIES.add("Transformer", implemented=True, description="Transformer / self-attention architectures")
