"""Tests for soft_prompt_toolkit.types — pure data containers."""

import numpy as np
import torch

from soft_prompt_toolkit import (
    DictionaryDiagnostics,
    GreedyStep,
    LogprobCache,
    Mixture,
    SupportMetrics,
)


def test_logprob_cache_construction():
    """LogprobCache holds its fields correctly."""
    cache = LogprobCache(
        prompts=["a", "b"],
        test_inputs=["x"],
        shared_indices=torch.tensor([0, 1, 2]),
        logprobs=torch.randn(2, 1, 3),
    )
    assert cache.prompts == ["a", "b"]
    assert cache.logprobs.shape == (2, 1, 3)
    assert cache.shared_indices.dtype == torch.int64


def test_mixture_construction():
    """Mixture pairs a cache with a weight vector."""
    cache = LogprobCache(
        prompts=["a"],
        test_inputs=["x"],
        shared_indices=torch.tensor([0]),
        logprobs=torch.randn(1, 1, 1),
    )
    m = Mixture(cache=cache, weights=np.array([1.0]))
    assert m.weights.shape == (1,)
    assert m.cache is cache


def test_support_metrics_fields():
    sm = SupportMetrics(precision=0.8, recall=0.6, f1=0.685, true_size=5, recovered_size=4)
    assert sm.precision == 0.8
    assert sm.true_size == 5


def test_greedy_step_fields():
    gs = GreedyStep(step=1, prompt_idx=3, prompt="hello", kl=0.5, weights=np.array([1.0]))
    assert gs.step == 1
    assert gs.prompt == "hello"


def test_dictionary_diagnostics_fields():
    dd = DictionaryDiagnostics(
        singular_values=np.array([3.0, 2.0, 1.0]),
        condition_number=3.0,
        effective_rank=2.5,
        mean_cosine=0.7,
    )
    assert dd.condition_number == 3.0
    assert len(dd.singular_values) == 3
