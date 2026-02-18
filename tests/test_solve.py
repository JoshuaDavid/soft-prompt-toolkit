"""Tests for soft_prompt_toolkit.solve — weight optimization."""

import warnings

import numpy as np
import pytest
import torch

from soft_prompt_toolkit import (
    LogprobCache,
    Mixture,
    find_weights,
    greedy_select,
    make_synthetic_target,
    mixture_mean_kl,
)


class TestFindWeights:
    def test_returns_mixture(self, synthetic_cache, synthetic_target):
        target, _ = synthetic_target
        mixture = find_weights(synthetic_cache, target, alpha=0.01, device="cpu")
        assert isinstance(mixture, Mixture)
        assert mixture.weights.shape == (20,)

    def test_reduces_kl(self, synthetic_cache, synthetic_target):
        """Solver should produce lower KL than uniform weights."""
        target, _ = synthetic_target
        mixture = find_weights(synthetic_cache, target, alpha=0.01, device="cpu")
        kl_solved = mixture_mean_kl(mixture, target)
        uniform = Mixture(cache=synthetic_cache, weights=np.ones(20) / 20)
        kl_uniform = mixture_mean_kl(uniform, target)
        assert kl_solved < kl_uniform

    def test_recovers_sparse_support(self, synthetic_cache, synthetic_target):
        """With enough data and low alpha, solver should find the active prompts."""
        target, true_w = synthetic_target
        mixture = find_weights(synthetic_cache, target, alpha=0.001, device="cpu")
        # The 3 true active prompts (3, 7, 12) should have largest |weights|
        active_true = set(np.where(np.abs(true_w) > 0.01)[0])
        top_recovered = set(np.argsort(-np.abs(mixture.weights))[:3])
        assert active_true == top_recovered

    def test_cpu_large_problem_warns(self):
        """Warn when solving on CPU with N*V > 1M."""
        torch.manual_seed(0)
        K, N, V = 3, 100, 20000  # N*V = 2M
        lp = torch.randn(K, N, V)
        lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
        cache = LogprobCache(
            prompts=["a", "b", "c"],
            test_inputs=[f"i{i}" for i in range(N)],
            shared_indices=torch.arange(V),
            logprobs=lp,
        )
        target = lp[0]  # just use first prompt as target
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            find_weights(cache, target, alpha=0.01, device="cpu")
            cpu_warnings = [x for x in w if "CPU" in str(x.message)]
            assert len(cpu_warnings) == 1

    def test_higher_alpha_sparser(self, synthetic_cache, synthetic_target):
        """Higher alpha should produce sparser weights (more near-zero)."""
        target, _ = synthetic_target
        m_low = find_weights(synthetic_cache, target, alpha=0.001, device="cpu")
        m_high = find_weights(synthetic_cache, target, alpha=0.1, device="cpu")
        nnz_low = np.sum(np.abs(m_low.weights) > 0.01)
        nnz_high = np.sum(np.abs(m_high.weights) > 0.01)
        assert nnz_high <= nnz_low


class TestGreedySelect:
    def test_returns_steps(self, synthetic_cache, synthetic_target):
        target, _ = synthetic_target
        steps = greedy_select(
            synthetic_cache, target, alpha=0.01, max_steps=3, verbose=False
        )
        assert len(steps) == 3
        assert steps[0].step == 1
        assert steps[1].step == 2

    def test_kl_monotonically_decreasing(self, synthetic_cache, synthetic_target):
        """Each greedy step should decrease or maintain KL."""
        target, _ = synthetic_target
        steps = greedy_select(
            synthetic_cache, target, alpha=0.01, max_steps=5, verbose=False
        )
        kls = [s.kl for s in steps]
        for i in range(1, len(kls)):
            assert kls[i] <= kls[i - 1] + 1e-6

    def test_first_step_picks_best_single(self, synthetic_cache, synthetic_target):
        """First greedy pick should be the single prompt with lowest solo KL."""
        target, true_w = synthetic_target
        steps = greedy_select(
            synthetic_cache, target, alpha=0.01, max_steps=1, verbose=False
        )
        # Prompt 3 has the largest true weight (1.5), should be picked first
        assert steps[0].prompt_idx == 3

    def test_early_stopping(self, synthetic_cache, synthetic_target):
        """With aggressive early_stop_rtol, should stop before max_steps."""
        target, _ = synthetic_target
        steps = greedy_select(
            synthetic_cache,
            target,
            alpha=0.01,
            max_steps=20,
            early_stop_rtol=0.5,  # very aggressive
            verbose=False,
        )
        assert len(steps) < 20

    def test_weights_grow_with_steps(self, synthetic_cache, synthetic_target):
        """Weight vector length should equal step number."""
        target, _ = synthetic_target
        steps = greedy_select(
            synthetic_cache, target, alpha=0.01, max_steps=4, verbose=False
        )
        for s in steps:
            assert len(s.weights) == s.step
