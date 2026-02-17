"""Internal utilities shared across modules."""

from __future__ import annotations

import torch.nn as nn
from transformers import PreTrainedModel


def get_embed_layer(model: PreTrainedModel) -> nn.Embedding:
    """Get the token embedding layer from a HuggingFace causal LM.

    Supports multiple architectures:

    - LLaMA / Qwen / Mistral: ``model.model.embed_tokens``
    - GPT-2 / GPT-Neo / GPT-J: ``model.transformer.wte``
    - OPT: ``model.model.decoder.embed_tokens``
    - BLOOM: ``model.transformer.word_embeddings``
    - Falcon: ``model.transformer.word_embeddings``
    - Gemma: ``model.model.embed_tokens``

    Raises:
        AttributeError: If the model architecture is not recognized.
    """
    # Try common attribute paths in order of popularity
    for path in [
        ("model", "embed_tokens"),
        ("transformer", "wte"),
        ("model", "decoder", "embed_tokens"),
        ("transformer", "word_embeddings"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.Embedding):
                return obj
        except AttributeError:
            continue

    raise AttributeError(
        f"Cannot find embedding layer for {type(model).__name__}. "
        f"Supported architectures: LLaMA, Qwen, Mistral, GPT-2, GPT-Neo, "
        f"GPT-J, OPT, BLOOM, Falcon, Gemma."
    )
