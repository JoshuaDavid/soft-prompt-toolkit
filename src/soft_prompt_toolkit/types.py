"""Pure data containers for the soft prompt toolkit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class LogprobCache:
    """Cached next-token logprobs for a set of prompts on a shared vocabulary.

    The core data artifact: once built, all downstream operations
    (solving, evaluation, subsetting) work purely on these tensors.

    Attributes:
        prompts: The K dictionary prompt strings.
        test_inputs: The N test input prefix strings.
        shared_indices: Token IDs in the shared vocabulary subset.
        logprobs: Log-probability tensor, shape ``[K, N, V]``.
    """

    prompts: list[str]
    test_inputs: list[str]
    shared_indices: Int[Tensor, " V"]
    logprobs: Float[Tensor, "K N V"]


@dataclass
class Mixture:
    """A LogprobCache paired with a weight vector: a Product-of-Experts distribution.

    The combined log-probability under this mixture is::

        log p_combined(t|x) = sum_i w_i * log p_i(t|x) - log Z(x)

    where ``Z(x)`` normalizes over tokens ``t`` for each input ``x``.

    Attributes:
        cache: The underlying logprob data.
        weights: Sparse weight vector of length K.
    """

    cache: LogprobCache
    weights: Float[np.ndarray, " K"]


@dataclass
class DictionaryDiagnostics:
    """Result of dictionary conditioning analysis.

    Attributes:
        singular_values: Singular values of the dictionary matrix A.
        condition_number: max(s) / min(s) of the SVD.
        effective_rank: Shannon-entropy-based effective rank.
        mean_cosine: Mean pairwise cosine similarity between prompts.
    """

    singular_values: Float[np.ndarray, " R"]
    condition_number: float
    effective_rank: float
    mean_cosine: float


@dataclass
class SupportMetrics:
    """Precision/recall/F1 for active prompt recovery.

    Attributes:
        precision: Fraction of recovered active prompts that are truly active.
        recall: Fraction of truly active prompts that were recovered.
        f1: Harmonic mean of precision and recall.
        true_size: Number of truly active prompts.
        recovered_size: Number of recovered active prompts.
    """

    precision: float
    recall: float
    f1: float
    true_size: int
    recovered_size: int


@dataclass
class GreedyStep:
    """One step of greedy forward selection.

    Attributes:
        step: The 1-indexed step number.
        prompt_idx: Index of the prompt added at this step.
        prompt: The prompt string added at this step.
        kl: KL divergence after adding this prompt.
        weights: Weight vector at this step (length = step).
    """

    step: int
    prompt_idx: int
    prompt: str
    kl: float
    weights: Float[np.ndarray, " S"]
