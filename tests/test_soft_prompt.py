"""Tests for soft_prompt_toolkit.soft_prompt — SoftPrompt module and training."""

import warnings

import numpy as np
import pytest
import torch

from soft_prompt_toolkit import (
    Mixture,
    SoftPrompt,
    collect_dictionary,
    evaluate_soft_prompt,
    find_weights,
    generate,
    make_synthetic_target,
    mixture_mean_kl,
    soft_prompt_from_text,
    train_residual,
    train_soft_prompt,
    train_soft_prompt_to_distribution,
)
from soft_prompt_toolkit._utils import get_embed_layer


# ── SoftPrompt module ────────────────────────────────────────────────────


class TestSoftPrompt:
    def test_shape_default_init(self):
        sp = SoftPrompt(10, 256)
        assert sp().shape == (10, 256)

    def test_shape_custom_init(self):
        init = torch.randn(5, 128)
        sp = SoftPrompt(5, 128, init_embeddings=init)
        assert sp().shape == (5, 128)

    def test_init_embeddings_not_shared(self):
        """Modifying the init tensor shouldn't affect the SoftPrompt."""
        init = torch.randn(3, 64)
        sp = SoftPrompt(3, 64, init_embeddings=init)
        init.fill_(0)
        assert sp().abs().sum() > 0

    def test_prepend_to(self):
        sp = SoftPrompt(4, 32)
        inputs = torch.randn(2, 10, 32)
        out = sp.prepend_to(inputs)
        assert out.shape == (2, 14, 32)

    def test_prepend_dtype_matching(self):
        """prepend_to should match the dtype of the input embeddings."""
        sp = SoftPrompt(4, 32)  # float32 by default
        inputs = torch.randn(1, 5, 32, dtype=torch.float16)
        out = sp.prepend_to(inputs)
        assert out.dtype == torch.float16

    def test_is_differentiable(self):
        sp = SoftPrompt(3, 16)
        out = sp()
        loss = out.sum()
        loss.backward()
        assert sp.prompt_embeddings.grad is not None


# ── get_embed_layer ──────────────────────────────────────────────────────


class TestGetEmbedLayer:
    def test_gpt_neo(self, tiny_model):
        embed = get_embed_layer(tiny_model)
        assert isinstance(embed, torch.nn.Embedding)
        assert embed.weight.shape == (50257, 256)

    def test_unknown_model_raises(self):
        """Should raise AttributeError for unrecognized models."""
        import torch.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(10, 10)

        # Pretend it's a PreTrainedModel
        fake = FakeModel()
        with pytest.raises(AttributeError, match="Cannot find embedding layer"):
            get_embed_layer(fake)


# ── soft_prompt_from_text ────────────────────────────────────────────────


class TestSoftPromptFromText:
    def test_shape(self, tiny_model, tiny_tokenizer):
        sp = soft_prompt_from_text("Hello world", tiny_model, tiny_tokenizer)
        T = sp().shape[0]
        assert T > 0
        assert sp().shape[1] == 256  # hidden_size

    def test_num_tokens_override(self, tiny_model, tiny_tokenizer):
        sp = soft_prompt_from_text(
            "Hello world", tiny_model, tiny_tokenizer, num_tokens=10
        )
        assert sp().shape[0] == 10

    def test_truncation(self, tiny_model, tiny_tokenizer):
        sp = soft_prompt_from_text(
            "Hello world", tiny_model, tiny_tokenizer, num_tokens=1
        )
        assert sp().shape[0] == 1


# ── train_soft_prompt ────────────────────────────────────────────────────


class TestTrainSoftPrompt:
    def test_loss_decreases(self, tiny_model, tiny_tokenizer):
        """Training loss should decrease over epochs."""
        texts = [
            "Once upon a time there was a little cat who loved to play with balls and run around in the garden",
            "The big dog ran across the green field and jumped over the tall fence making a loud sound as it went",
            "A small bird sat on the long tree branch and sang a beautiful song for everyone to hear in the morning",
        ]
        sp, losses = train_soft_prompt(
            tiny_model,
            tiny_tokenizer,
            texts,
            num_tokens=3,
            epochs=3,
            lr=3e-2,
            batch_size=2,
            max_seq_len=40,
            verbose=False,
        )
        assert len(losses) == 3
        assert losses[-1] < losses[0]
        assert sp().shape == (3, 256)

    def test_init_from_text(self, tiny_model, tiny_tokenizer):
        texts = [
            "Once upon a time there was a little cat who loved to play with balls and run around in the garden",
        ]
        sp, losses = train_soft_prompt(
            tiny_model,
            tiny_tokenizer,
            texts,
            num_tokens=3,
            epochs=2,
            lr=1e-2,
            batch_size=1,
            max_seq_len=40,
            init_text="Once upon",
            verbose=False,
        )
        assert sp().shape == (3, 256)

    def test_short_texts_raises(self, tiny_model, tiny_tokenizer):
        """Texts too short to have 20 tokens should raise ValueError."""
        with pytest.raises(ValueError, match="No training examples"):
            train_soft_prompt(
                tiny_model,
                tiny_tokenizer,
                ["Hi"],
                num_tokens=3,
                epochs=1,
                verbose=False,
            )

    def test_short_texts_warns(self, tiny_model, tiny_tokenizer):
        """Should warn when some examples are dropped due to short length."""
        texts = [
            "Hi",  # too short
            "Once upon a time there was a little cat who loved to play with balls and run around in the garden",
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp, losses = train_soft_prompt(
                tiny_model,
                tiny_tokenizer,
                texts,
                num_tokens=3,
                epochs=1,
                verbose=False,
            )
            drop_warnings = [x for x in w if "Dropped" in str(x.message)]
            assert len(drop_warnings) == 1
            assert "1/2" in str(drop_warnings[0].message)

    def test_custom_min_tokens(self, tiny_model, tiny_tokenizer):
        """min_tokens parameter should control the filtering threshold."""
        texts = [
            "Once upon a time there was a little cat who loved to play",  # ~14 tokens
        ]
        # Default min_tokens=20 would drop this
        with pytest.raises(ValueError, match="No training examples"):
            train_soft_prompt(
                tiny_model, tiny_tokenizer, texts,
                num_tokens=3, epochs=1, verbose=False,
            )
        # But min_tokens=5 should keep it
        sp, losses = train_soft_prompt(
            tiny_model, tiny_tokenizer, texts,
            num_tokens=3, epochs=1, min_tokens=5, verbose=False,
        )
        assert len(losses) == 1


# ── train_soft_prompt_to_distribution ────────────────────────────────────


class TestTrainToDistribution:
    def test_runs_and_returns_soft_prompt(self, tiny_model, tiny_tokenizer, tiny_cache):
        target = make_synthetic_target(tiny_cache, np.array([1.0, 0.5, 0.2]))
        sp = train_soft_prompt_to_distribution(
            tiny_model,
            tiny_tokenizer,
            target,
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            num_tokens=3,
            epochs=2,
            lr=1e-2,
            batch_size=3,
            device="cpu",
            verbose=False,
        )
        assert sp().shape[0] == 3
        assert sp().shape[1] == 256


# ── train_residual ───────────────────────────────────────────────────────


class TestTrainResidual:
    def test_runs_and_returns_components(
        self, tiny_model, tiny_tokenizer, tiny_cache
    ):
        target = make_synthetic_target(tiny_cache, np.array([1.0, 0.5, 0.2]))
        mixture = find_weights(tiny_cache, target, alpha=0.01, device="cpu")

        res_sp, w_new, losses = train_residual(
            tiny_model,
            tiny_tokenizer,
            mixture,
            target,
            tiny_cache.test_inputs,
            tiny_cache.shared_indices,
            num_tokens=2,
            epochs=2,
            lr=1e-2,
            batch_size=3,
            device="cpu",
            verbose=False,
        )
        assert isinstance(res_sp, SoftPrompt)
        assert isinstance(w_new, float)
        assert len(losses) == 2


# ── generate ─────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_strings(self, tiny_model, tiny_tokenizer):
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        samples = generate(
            tiny_model,
            tiny_tokenizer,
            sp,
            max_new_tokens=5,
            num_samples=2,
            temperature=0.5,
        )
        assert len(samples) == 2
        assert all(isinstance(s, str) for s in samples)

    def test_with_prefix(self, tiny_model, tiny_tokenizer):
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        samples = generate(
            tiny_model,
            tiny_tokenizer,
            sp,
            prefix="Once",
            max_new_tokens=5,
            num_samples=1,
            temperature=0.5,
        )
        assert len(samples) == 1

    def test_greedy_deterministic(self, tiny_model, tiny_tokenizer):
        """temperature=0 should give greedy (argmax) decoding, deterministic."""
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        s1 = generate(
            tiny_model, tiny_tokenizer, sp,
            prefix="Once", max_new_tokens=10, num_samples=1, temperature=0,
        )
        s2 = generate(
            tiny_model, tiny_tokenizer, sp,
            prefix="Once", max_new_tokens=10, num_samples=1, temperature=0,
        )
        assert s1 == s2


# ── evaluate_soft_prompt ─────────────────────────────────────────────────


class TestEvaluateSoftPrompt:
    def test_returns_float(self, tiny_model, tiny_tokenizer):
        texts = [
            "Once upon a time there was a little cat who loved to play with balls and run around in the garden",
        ]
        sp = SoftPrompt(3, tiny_model.config.hidden_size)
        loss = evaluate_soft_prompt(tiny_model, tiny_tokenizer, sp, texts)
        assert isinstance(loss, float)
        assert loss > 0

    def test_trained_lower_than_random(self, tiny_model, tiny_tokenizer):
        """A trained soft prompt should have lower eval loss than a random one."""
        texts = [
            "Once upon a time there was a little cat who loved to play with balls and run around in the garden",
            "The big dog ran across the green field and jumped over the tall fence making a loud sound as it went",
            "A small bird sat on the long tree branch and sang a beautiful song for everyone to hear in the morning",
        ]
        sp_trained, _ = train_soft_prompt(
            tiny_model, tiny_tokenizer, texts,
            num_tokens=3, epochs=3, lr=3e-2, batch_size=2, max_seq_len=40,
            verbose=False,
        )
        sp_random = SoftPrompt(3, tiny_model.config.hidden_size)

        loss_trained = evaluate_soft_prompt(tiny_model, tiny_tokenizer, sp_trained, texts)
        loss_random = evaluate_soft_prompt(tiny_model, tiny_tokenizer, sp_random, texts)
        assert loss_trained < loss_random
