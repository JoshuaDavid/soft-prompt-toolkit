"""Shared fixtures for soft_prompt_toolkit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_prompt_toolkit import LogprobCache, Mixture, collect_dictionary

MODEL_NAME = "roneneldan/TinyStories-8M"


@pytest.fixture(scope="session")
def tiny_model():
    """Load TinyStories-8M model (session-scoped, shared across all tests)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Load TinyStories-8M tokenizer (session-scoped)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def tiny_cache(tiny_model, tiny_tokenizer):
    """Collect a small LogprobCache with TinyStories-8M (session-scoped).

    3 prompts x 3 test inputs, top_k=50.
    """
    prompts = ["Once upon a time", "The cat sat on", "A little girl named"]
    test_inputs = ["The", "Once", "In"]
    cache = collect_dictionary(
        tiny_model,
        tiny_tokenizer,
        prompts,
        test_inputs,
        top_k=50,
        device="cpu",
        verbose=False,
    )
    return cache


@pytest.fixture
def synthetic_cache():
    """A deterministic synthetic LogprobCache for unit tests (no model needed)."""
    torch.manual_seed(42)
    K, N, V = 20, 10, 500
    logprobs_raw = torch.randn(K, N, V)
    logprobs = logprobs_raw - torch.logsumexp(logprobs_raw, dim=-1, keepdim=True)
    return LogprobCache(
        prompts=[f"prompt_{i}" for i in range(K)],
        test_inputs=[f"input_{i}" for i in range(N)],
        shared_indices=torch.arange(V, dtype=torch.long),
        logprobs=logprobs,
    )


@pytest.fixture
def synthetic_target(synthetic_cache):
    """A deterministic synthetic target with known sparse weights."""
    from soft_prompt_toolkit import make_synthetic_target

    true_w = np.zeros(20)
    true_w[3] = 1.5
    true_w[7] = 0.8
    true_w[12] = -0.3
    target = make_synthetic_target(synthetic_cache, true_w)
    return target, true_w
