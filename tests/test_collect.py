"""Tests for soft_prompt_toolkit.collect — logprob collection and cache utilities."""

import os
import tempfile

import numpy as np
import pytest
import torch

from soft_prompt_toolkit import (
    LogprobCache,
    SoftPrompt,
    cap_vocab,
    collect_dictionary,
    collect_target_hard,
    collect_target_soft,
    load_cache,
    make_synthetic_target,
    renormalize,
    sample_cache,
    save_cache,
    subset_cache,
)


# ── cap_vocab ────────────────────────────────────────────────────────────


class TestCapVocab:
    def test_no_cap_when_under_limit(self):
        """If V <= max_vocab, returns identity indices."""
        lp = torch.randn(5, 3, 100)
        idx = cap_vocab(lp, max_vocab=200)
        assert idx.shape[0] == 100
        assert torch.equal(idx, torch.arange(100))

    def test_caps_to_max_vocab(self):
        """Returns exactly max_vocab indices when V > max_vocab."""
        lp = torch.randn(5, 3, 500)
        idx = cap_vocab(lp, max_vocab=100)
        assert idx.shape[0] == 100

    def test_indices_sorted(self):
        """Returned indices are sorted."""
        lp = torch.randn(5, 3, 500)
        idx = cap_vocab(lp, max_vocab=100)
        assert torch.all(idx[1:] >= idx[:-1])

    def test_with_target(self):
        """Works with target_logprobs provided."""
        lp = torch.randn(5, 3, 500)
        target = torch.randn(3, 500)
        idx = cap_vocab(lp, max_vocab=100, target_logprobs=target)
        assert idx.shape[0] == 100


# ── make_synthetic_target ────────────────────────────────────────────────


class TestMakeSyntheticTarget:
    def test_output_normalized(self, synthetic_cache):
        target = make_synthetic_target(synthetic_cache, np.ones(20))
        sums = torch.exp(target).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_numpy_scalar_dict_keys(self, synthetic_cache):
        """Dict with np.int64 keys and np.float64 values should work."""
        weights = {np.int64(0): np.float64(1.0), np.int64(5): np.float64(0.5)}
        target = make_synthetic_target(synthetic_cache, weights)
        assert target.shape == (10, 500)

    def test_dict_weights(self, synthetic_cache):
        """Accepts dict[int, float] as weights."""
        target = make_synthetic_target(synthetic_cache, {0: 1.0, 5: 0.5})
        assert target.shape == (10, 500)
        sums = torch.exp(target).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_noise_changes_output(self, synthetic_cache):
        t1 = make_synthetic_target(synthetic_cache, np.ones(20), noise_sigma=0.0)
        t2 = make_synthetic_target(synthetic_cache, np.ones(20), noise_sigma=0.1)
        assert not torch.allclose(t1, t2)
        # But noisy output is still normalized
        sums = torch.exp(t2).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_deterministic_with_seed(self, synthetic_cache):
        t1 = make_synthetic_target(synthetic_cache, np.ones(20), noise_sigma=0.1, seed=7)
        t2 = make_synthetic_target(synthetic_cache, np.ones(20), noise_sigma=0.1, seed=7)
        assert torch.allclose(t1, t2)


# ── subset_cache, sample_cache ───────────────────────────────────────────


class TestCacheOps:
    def test_subset_cache(self, synthetic_cache):
        sub = subset_cache(synthetic_cache, [0, 5, 10])
        assert sub.logprobs.shape[0] == 3
        assert len(sub.prompts) == 3
        assert sub.prompts[0] == "prompt_0"
        assert sub.prompts[1] == "prompt_5"
        assert torch.equal(sub.logprobs[0], synthetic_cache.logprobs[0])

    def test_sample_cache_size(self, synthetic_cache):
        sampled = sample_cache(synthetic_cache, 5, seed=42)
        assert sampled.logprobs.shape[0] == 5
        assert len(sampled.prompts) == 5

    def test_sample_cache_deterministic(self, synthetic_cache):
        s1 = sample_cache(synthetic_cache, 5, seed=42)
        s2 = sample_cache(synthetic_cache, 5, seed=42)
        assert s1.prompts == s2.prompts
        assert torch.equal(s1.logprobs, s2.logprobs)

    def test_sample_cache_different_seeds(self, synthetic_cache):
        s1 = sample_cache(synthetic_cache, 5, seed=42)
        s2 = sample_cache(synthetic_cache, 5, seed=99)
        assert s1.prompts != s2.prompts


# ── save_cache, load_cache ───────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, synthetic_cache):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_cache(synthetic_cache, path)
            loaded = load_cache(path)
            assert loaded.prompts == synthetic_cache.prompts
            assert loaded.test_inputs == synthetic_cache.test_inputs
            assert torch.equal(loaded.shared_indices, synthetic_cache.shared_indices)
            assert torch.allclose(loaded.logprobs, synthetic_cache.logprobs)
        finally:
            os.unlink(path)


# ── collect_dictionary (with model) ──────────────────────────────────────


class TestCollectDictionary:
    def test_shape(self, tiny_cache):
        """Cache has correct K, N, V dimensions."""
        K, N, V = tiny_cache.logprobs.shape
        assert K == 3
        assert N == 3
        assert V > 0  # shared vocab discovered

    def test_prompts_preserved(self, tiny_cache):
        assert tiny_cache.prompts == [
            "Once upon a time",
            "The cat sat on",
            "A little girl named",
        ]

    def test_logprobs_finite(self, tiny_cache):
        """All logprobs should be finite (no nan/inf)."""
        assert torch.isfinite(tiny_cache.logprobs).all()


# ── collect_target_hard ──────────────────────────────────────────────────


class TestCollectTargetHard:
    def test_shape(self, tiny_model, tiny_tokenizer, tiny_cache):
        target = collect_target_hard(
            tiny_model,
            tiny_tokenizer,
            "There was a",
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            device="cpu",
        )
        assert target.shape == (3, tiny_cache.shared_indices.shape[0])

    def test_finite(self, tiny_model, tiny_tokenizer, tiny_cache):
        target = collect_target_hard(
            tiny_model,
            tiny_tokenizer,
            "There was a",
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            device="cpu",
        )
        assert torch.isfinite(target).all()


# ── collect_target_soft ──────────────────────────────────────────────────


class TestCollectTargetSoft:
    def test_shape(self, tiny_model, tiny_tokenizer, tiny_cache):
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        target = collect_target_soft(
            tiny_model,
            tiny_tokenizer,
            sp,
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            device="cpu",
        )
        assert target.shape == (3, tiny_cache.shared_indices.shape[0])

    def test_finite(self, tiny_model, tiny_tokenizer, tiny_cache):
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        target = collect_target_soft(
            tiny_model,
            tiny_tokenizer,
            sp,
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            device="cpu",
        )
        assert torch.isfinite(target).all()
