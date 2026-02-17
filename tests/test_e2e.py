"""End-to-end tests: full pipeline flows with snapshot/regression values.

These tests exercise entire workflows and verify outputs against known-good
values from the initial sanity-check run. They serve as regression tests:
if an internal change accidentally alters numeric behavior, these will catch it.
"""

import numpy as np
import pytest
import torch

from soft_prompt_toolkit import (
    LogprobCache,
    Mixture,
    SoftPrompt,
    cap_vocab,
    collect_dictionary,
    collect_target_hard,
    collect_target_soft,
    condition_number,
    find_weights,
    greedy_select,
    kl_divergence,
    make_synthetic_target,
    mixture_kl,
    mixture_logprobs,
    mixture_mean_kl,
    mixture_summary,
    mixture_support,
    pairwise_cosine,
    renormalize,
    soft_prompt_from_text,
    support_recovery,
    top_k_agreement,
)


# ── E2E 1: Synthetic sparse recovery (no model) ─────────────────────────


class TestE2ESyntheticRecovery:
    """Full synthetic recovery: build cache, create target, solve, evaluate."""

    @pytest.fixture
    def pipeline_results(self):
        """Run the full synthetic recovery pipeline once, share results."""
        torch.manual_seed(42)
        K, N, V = 20, 10, 500
        logprobs_raw = torch.randn(K, N, V)
        logprobs = logprobs_raw - torch.logsumexp(logprobs_raw, dim=-1, keepdim=True)
        cache = LogprobCache(
            prompts=[f"prompt_{i}" for i in range(K)],
            test_inputs=[f"input_{i}" for i in range(N)],
            shared_indices=torch.arange(V, dtype=torch.long),
            logprobs=logprobs,
        )
        true_w = np.zeros(K)
        true_w[3] = 1.5
        true_w[7] = 0.8
        true_w[12] = -0.3
        target = make_synthetic_target(cache, true_w)
        mixture = find_weights(cache, target, alpha=0.001, device="cpu")
        return cache, target, true_w, mixture

    def test_mean_kl_regression(self, pipeline_results):
        """Mean KL matches snapshot value."""
        _, target, _, mixture = pipeline_results
        kl = mixture_mean_kl(mixture, target)
        assert abs(kl - 0.016556) < 0.005

    def test_support_recovery_recall(self, pipeline_results):
        """All 3 true active prompts are recovered."""
        _, _, true_w, mixture = pipeline_results
        sr = support_recovery(true_w, mixture.weights)
        assert sr.recall == 1.0

    def test_top_3_weights_are_correct_prompts(self, pipeline_results):
        """The 3 largest |weight| prompts are indices 3, 7, 12."""
        _, _, _, mixture = pipeline_results
        top3 = set(np.argsort(-np.abs(mixture.weights))[:3])
        assert top3 == {3, 7, 12}

    def test_weight_signs_match(self, pipeline_results):
        """Recovered weights have correct signs for the active prompts."""
        _, _, true_w, mixture = pipeline_results
        for idx in [3, 7, 12]:
            assert np.sign(mixture.weights[idx]) == np.sign(true_w[idx])

    def test_weight_magnitudes_close(self, pipeline_results):
        """Recovered weight magnitudes are within 10% of true values."""
        _, _, true_w, mixture = pipeline_results
        for idx in [3, 7, 12]:
            ratio = abs(mixture.weights[idx] / true_w[idx])
            assert 0.8 < ratio < 1.2, f"idx={idx}: ratio={ratio}"


class TestE2EGreedy:
    """End-to-end greedy forward selection."""

    @pytest.fixture
    def greedy_results(self):
        torch.manual_seed(42)
        K, N, V = 20, 10, 500
        logprobs_raw = torch.randn(K, N, V)
        logprobs = logprobs_raw - torch.logsumexp(logprobs_raw, dim=-1, keepdim=True)
        cache = LogprobCache(
            prompts=[f"prompt_{i}" for i in range(K)],
            test_inputs=[f"input_{i}" for i in range(N)],
            shared_indices=torch.arange(V, dtype=torch.long),
            logprobs=logprobs,
        )
        true_w = np.zeros(K)
        true_w[3] = 1.5
        true_w[7] = 0.8
        true_w[12] = -0.3
        target = make_synthetic_target(cache, true_w)
        steps = greedy_select(cache, target, alpha=0.01, max_steps=5, verbose=False)
        return steps

    def test_first_pick_is_prompt_3(self, greedy_results):
        """Prompt 3 (highest weight=1.5) should be selected first."""
        assert greedy_results[0].prompt_idx == 3

    def test_kl_decreasing(self, greedy_results):
        kls = [s.kl for s in greedy_results]
        for i in range(1, len(kls)):
            assert kls[i] <= kls[i - 1] + 1e-6

    def test_step_1_kl_regression(self, greedy_results):
        assert abs(greedy_results[0].kl - 0.4027) < 0.01

    def test_step_3_kl_regression(self, greedy_results):
        """After adding prompts 3, 7, 12, KL should drop significantly."""
        assert greedy_results[2].kl < 0.15


# ── E2E 2: Diagnostics ──────────────────────────────────────────────────


class TestE2EDiagnostics:
    @pytest.fixture
    def cache(self):
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

    def test_condition_number_regression(self, cache):
        diag = condition_number(cache)
        assert abs(diag.condition_number - 31.61) < 1.0

    def test_effective_rank_regression(self, cache):
        diag = condition_number(cache)
        assert abs(diag.effective_rank - 1.18) < 0.2

    def test_mean_cosine_regression(self, cache):
        diag = condition_number(cache)
        assert abs(diag.mean_cosine - 0.978) < 0.01


# ── E2E 3: Cap vocab + solve ────────────────────────────────────────────


class TestE2ECapVocab:
    def test_capped_solve(self):
        """Solving on capped vocabulary still produces low KL."""
        torch.manual_seed(42)
        K, N, V = 20, 10, 500
        logprobs_raw = torch.randn(K, N, V)
        logprobs = logprobs_raw - torch.logsumexp(logprobs_raw, dim=-1, keepdim=True)
        cache = LogprobCache(
            prompts=[f"prompt_{i}" for i in range(K)],
            test_inputs=[f"input_{i}" for i in range(N)],
            shared_indices=torch.arange(V, dtype=torch.long),
            logprobs=logprobs,
        )
        true_w = np.zeros(K)
        true_w[3] = 1.5
        true_w[7] = 0.8
        true_w[12] = -0.3
        target = make_synthetic_target(cache, true_w)

        capped_idx = cap_vocab(logprobs, max_vocab=100, target_logprobs=target)
        capped_logprobs = logprobs[:, :, capped_idx]
        capped_target = target[:, capped_idx]
        capped_cache = LogprobCache(
            prompts=cache.prompts,
            test_inputs=cache.test_inputs,
            shared_indices=cache.shared_indices[capped_idx],
            logprobs=capped_logprobs,
        )
        mixture = find_weights(capped_cache, capped_target, alpha=0.001, device="cpu")
        kl = mixture_mean_kl(mixture, capped_target)
        assert kl < 0.05  # still good despite vocab reduction


# ── E2E 4: Full model pipeline (TinyStories-8M) ─────────────────────────


class TestE2EModelPipeline:
    """Full pipeline with a real model: collect, solve, evaluate."""

    @pytest.fixture(scope="class")
    def pipeline(self, tiny_model, tiny_tokenizer, tiny_cache):
        target = collect_target_hard(
            tiny_model,
            tiny_tokenizer,
            "There was a",
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            device="cpu",
        )
        mixture = find_weights(tiny_cache, target, alpha=0.01, device="cpu")
        return tiny_cache, target, mixture

    def test_shared_vocab_size(self, pipeline):
        cache, _, _ = pipeline
        assert cache.shared_indices.shape[0] == 265

    def test_mean_kl_regression(self, pipeline):
        _, target, mixture = pipeline
        kl = mixture_mean_kl(mixture, target)
        assert abs(kl - 0.3387) < 0.01

    def test_weights_regression(self, pipeline):
        _, _, mixture = pipeline
        w = mixture.weights
        assert abs(w[0] - 0.3102) < 0.02
        assert abs(w[1] - 0.3609) < 0.02
        assert abs(w[2] - 0.2062) < 0.02

    def test_all_weights_positive(self, pipeline):
        """For this target, all dictionary prompts should have positive weight."""
        _, _, mixture = pipeline
        assert (mixture.weights > 0).all()

    def test_summary_contains_prompts(self, pipeline):
        _, _, mixture = pipeline
        s = mixture_summary(mixture)
        assert "The cat sat on" in s

    def test_mixture_logprobs_normalized(self, pipeline):
        _, _, mixture = pipeline
        lp = mixture_logprobs(mixture)
        sums = torch.exp(lp).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-4)

    def test_greedy_first_pick(self, pipeline):
        cache, target, _ = pipeline
        steps = greedy_select(cache, target, alpha=0.01, max_steps=1, verbose=False)
        # Prompt 0 ("Once upon a time") should be first pick
        assert steps[0].prompt_idx == 0

    def test_soft_prompt_target(self, pipeline, tiny_model, tiny_tokenizer):
        """Collecting logprobs from a soft prompt works end-to-end."""
        cache, _, _ = pipeline
        sp = soft_prompt_from_text("Once upon", tiny_model, tiny_tokenizer)
        soft_target = collect_target_soft(
            tiny_model,
            tiny_tokenizer,
            sp,
            cache.test_inputs,
            cache.shared_indices,
            device="cpu",
        )
        assert soft_target.shape == (3, cache.shared_indices.shape[0])
        assert torch.isfinite(soft_target).all()

        # Solve against soft target
        sp_mixture = find_weights(cache, soft_target, alpha=0.01, device="cpu")
        sp_kl = mixture_mean_kl(sp_mixture, soft_target)
        assert abs(sp_kl - 0.1887) < 0.02
