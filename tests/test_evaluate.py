"""Tests for soft_prompt_toolkit.evaluate — pure tensor functions."""

import numpy as np
import pytest
import torch

from soft_prompt_toolkit import (
    LogprobCache,
    Mixture,
    condition_number,
    kl_divergence,
    mixture_kl,
    mixture_logprobs,
    mixture_mean_kl,
    mixture_summary,
    mixture_support,
    pairwise_cosine,
    renormalize,
    support_recovery,
    top_k_agreement,
)


# ── renormalize ──────────────────────────────────────────────────────────


class TestRenormalize:
    def test_sums_to_one(self):
        """exp(renormalize(x)).sum(-1) == 1 for arbitrary input."""
        x = torch.randn(5, 100)
        r = renormalize(x)
        sums = torch.exp(r).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5), atol=1e-5)

    def test_idempotent(self):
        """Applying renormalize twice gives the same result."""
        x = torch.randn(3, 50)
        r1 = renormalize(x)
        r2 = renormalize(r1)
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_already_normalized(self):
        """If input is already normalized, output is unchanged."""
        x = torch.randn(4, 30)
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        r = renormalize(x)
        assert torch.allclose(x, r, atol=1e-6)


# ── kl_divergence ────────────────────────────────────────────────────────


class TestKLDivergence:
    def test_self_kl_is_zero(self):
        """KL(p || p) == 0 for any distribution."""
        p = torch.randn(8, 200)
        p = p - torch.logsumexp(p, dim=-1, keepdim=True)
        kl = kl_divergence(p, p)
        assert torch.allclose(kl, torch.zeros(8), atol=1e-5)

    def test_nonnegative(self):
        """KL divergence is always >= 0."""
        torch.manual_seed(99)
        p = torch.randn(10, 100)
        q = torch.randn(10, 100)
        kl = kl_divergence(p, q)
        assert (kl >= -1e-6).all()

    def test_asymmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        torch.manual_seed(7)
        p = torch.randn(5, 50)
        q = torch.randn(5, 50)
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert not torch.allclose(kl_pq, kl_qp, atol=1e-4)

    def test_shape(self):
        """Output has shape [N] for input [N, V]."""
        kl = kl_divergence(torch.randn(7, 30), torch.randn(7, 30))
        assert kl.shape == (7,)


# ── mixture_logprobs ─────────────────────────────────────────────────────


class TestMixtureLogprobs:
    def test_output_normalized(self, synthetic_cache):
        """Mixture logprobs should sum to 1 after exp."""
        w = np.ones(20) / 20
        mixture = Mixture(cache=synthetic_cache, weights=w)
        lp = mixture_logprobs(mixture)
        sums = torch.exp(lp).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_shape(self, synthetic_cache):
        """Output shape is [N, V]."""
        w = np.zeros(20)
        w[0] = 1.0
        mixture = Mixture(cache=synthetic_cache, weights=w)
        lp = mixture_logprobs(mixture)
        assert lp.shape == (10, 500)

    def test_single_weight_recovers_input(self, synthetic_cache):
        """With w=[0,...,0,1,0,...,0], mixture_logprobs == renormalize(logprobs[i])."""
        w = np.zeros(20)
        w[5] = 1.0
        mixture = Mixture(cache=synthetic_cache, weights=w)
        lp = mixture_logprobs(mixture)
        expected = renormalize(synthetic_cache.logprobs[5])
        assert torch.allclose(lp, expected, atol=1e-5)


# ── mixture_kl, mixture_mean_kl ─────────────────────────────────────────


class TestMixtureKL:
    def test_nonnegative(self, synthetic_cache, synthetic_target):
        target, _ = synthetic_target
        w = np.ones(20) / 20
        mixture = Mixture(cache=synthetic_cache, weights=w)
        kl = mixture_kl(mixture, target)
        assert (kl >= -1e-6).all()

    def test_mean_kl_is_mean_of_kl(self, synthetic_cache, synthetic_target):
        target, _ = synthetic_target
        w = np.ones(20) / 20
        mixture = Mixture(cache=synthetic_cache, weights=w)
        kl_per = mixture_kl(mixture, target)
        mean_kl = mixture_mean_kl(mixture, target)
        assert abs(mean_kl - kl_per.mean().item()) < 1e-6


# ── mixture_summary, mixture_support ─────────────────────────────────────


class TestMixtureSummarySupport:
    def test_summary_format(self, synthetic_cache):
        w = np.zeros(20)
        w[0] = 1.0
        w[3] = -0.5
        mixture = Mixture(cache=synthetic_cache, weights=w)
        s = mixture_summary(mixture)
        assert "Mixture:" in s
        assert "L0" in s
        lines = s.strip().split("\n")
        assert len(lines) >= 2  # header + at least one active

    def test_support_sorted_by_abs_weight(self, synthetic_cache):
        w = np.zeros(20)
        w[0] = 0.3
        w[5] = -0.8
        w[10] = 0.1
        mixture = Mixture(cache=synthetic_cache, weights=w)
        support = mixture_support(mixture, threshold=0.01)
        abs_weights = [abs(wt) for _, wt in support]
        assert abs_weights == sorted(abs_weights, reverse=True)

    def test_support_respects_threshold(self, synthetic_cache):
        w = np.zeros(20)
        w[0] = 0.05
        w[1] = 0.005  # below threshold
        mixture = Mixture(cache=synthetic_cache, weights=w)
        support = mixture_support(mixture, threshold=0.01)
        assert len(support) == 1
        assert support[0][0] == "prompt_0"


# ── top_k_agreement ──────────────────────────────────────────────────────


class TestTopKAgreement:
    def test_self_agreement_is_one(self):
        """Agreement of a distribution with itself is 1.0."""
        p = torch.randn(5, 100)
        assert abs(top_k_agreement(p, p, k=10) - 1.0) < 1e-6

    def test_range_zero_to_one(self):
        """Agreement is always in [0, 1]."""
        p = torch.randn(5, 100)
        q = torch.randn(5, 100)
        a = top_k_agreement(p, q, k=10)
        assert 0.0 <= a <= 1.0


# ── support_recovery ─────────────────────────────────────────────────────


class TestSupportRecovery:
    def test_perfect_recovery(self):
        true = np.array([1.0, 0.0, 0.5, 0.0, -0.3])
        sr = support_recovery(true, true)
        assert sr.precision == 1.0
        assert sr.recall == 1.0
        assert sr.f1 > 0.99

    def test_partial_recovery(self):
        true = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
        recovered = np.array([1.0, 0.0, 0.0, 0.0, 0.5])
        sr = support_recovery(true, recovered)
        assert sr.precision == 0.5  # 1 of 2 recovered is correct
        assert sr.recall == 0.5  # 1 of 2 true is recovered
        assert sr.true_size == 2
        assert sr.recovered_size == 2

    def test_empty_both(self):
        true = np.zeros(5)
        recovered = np.zeros(5)
        sr = support_recovery(true, recovered)
        assert sr.precision == 1.0
        assert sr.recall == 1.0


# ── condition_number ─────────────────────────────────────────────────────


class TestConditionNumber:
    def test_returns_positive_values(self, synthetic_cache):
        diag = condition_number(synthetic_cache)
        assert diag.condition_number > 0
        assert diag.effective_rank > 0
        assert len(diag.singular_values) > 0

    def test_singular_values_descending(self, synthetic_cache):
        diag = condition_number(synthetic_cache)
        sv = diag.singular_values
        assert all(sv[i] >= sv[i + 1] - 1e-10 for i in range(len(sv) - 1))

    def test_effective_rank_bounded(self, synthetic_cache):
        """Effective rank is between 1 and K."""
        K = synthetic_cache.logprobs.shape[0]
        diag = condition_number(synthetic_cache)
        assert 1.0 <= diag.effective_rank <= K

    def test_vocab_capping(self, synthetic_cache):
        """Works with max_vocab smaller than V."""
        diag = condition_number(synthetic_cache, max_vocab=50)
        assert diag.condition_number > 0


# ── pairwise_cosine ──────────────────────────────────────────────────────


class TestPairwiseCosine:
    def test_diagonal_is_one(self, synthetic_cache):
        """Cosine similarity of a prompt with itself is 1.0."""
        pc = pairwise_cosine(synthetic_cache)
        diag = pc.diag()
        assert torch.allclose(diag, torch.ones(20), atol=1e-4)

    def test_symmetric(self, synthetic_cache):
        """Cosine matrix is symmetric."""
        pc = pairwise_cosine(synthetic_cache)
        assert torch.allclose(pc, pc.T, atol=1e-6)

    def test_shape(self, synthetic_cache):
        """Output shape is [K, K]."""
        pc = pairwise_cosine(synthetic_cache)
        assert pc.shape == (20, 20)

    def test_range(self, synthetic_cache):
        """Cosine values are in [-1, 1]."""
        pc = pairwise_cosine(synthetic_cache)
        assert pc.min() >= -1.0 - 1e-5
        assert pc.max() <= 1.0 + 1e-5
