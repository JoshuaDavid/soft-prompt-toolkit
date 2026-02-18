"""Functions that compute things from Mixtures and LogprobCaches.

All evaluation, metrics, and diagnostics live here. These are pure functions
on tensors — no model or tokenizer required.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor

from .types import DictionaryDiagnostics, LogprobCache, Mixture, SupportMetrics


@jaxtyped(typechecker=beartype)
def renormalize(logprobs: Float[Tensor, "*batch V"]) -> Float[Tensor, "*batch V"]:
    """Renormalize logprobs over the vocab dimension via logsumexp.

    Needed because logprobs are on a shared vocab subset, not the full
    vocabulary, so they don't necessarily sum to 1.
    """
    return logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)


@jaxtyped(typechecker=beartype)
def mixture_logprobs(mixture: Mixture) -> Float[Tensor, "N V"]:
    """Compute renormalized PoE combined logprobs from a Mixture."""
    w = torch.as_tensor(mixture.weights, dtype=torch.float32)
    combined = einsum(w, mixture.cache.logprobs, "K, K N V -> N V")
    return combined - torch.logsumexp(combined, dim=-1, keepdim=True)


@jaxtyped(typechecker=beartype)
def kl_divergence(
    p: Float[Tensor, "N V"],
    q: Float[Tensor, "N V"],
) -> Float[Tensor, " N"]:
    """KL(p || q) per row, with renormalization on the shared vocab subset.

    Both inputs are renormalized via logsumexp before computing KL.
    This is a thin wrapper — the renormalization is the only part that
    differs from a standard KL implementation.
    """
    p_norm = renormalize(p)
    q_norm = renormalize(q)
    return (p_norm.exp() * (p_norm - q_norm)).sum(dim=-1)


@jaxtyped(typechecker=beartype)
def mixture_kl(
    mixture: Mixture,
    target: Float[Tensor, "N V"],
) -> Float[Tensor, " N"]:
    """KL(target || mixture) per test input."""
    return kl_divergence(target, mixture_logprobs(mixture))


def mixture_mean_kl(mixture: Mixture, target: Float[Tensor, "N V"]) -> float:
    """Mean KL(target || mixture) across test inputs."""
    return mixture_kl(mixture, target).mean().item()


def mixture_summary(mixture: Mixture, top_n: int = 10) -> str:
    """Human-readable multi-line summary of a Mixture's active components."""
    w = mixture.weights
    l0 = int(np.sum(np.abs(w) > 0.01))
    l1 = float(np.sum(np.abs(w)))
    order = np.argsort(-np.abs(w))

    lines = [
        f"Mixture: {len(w)} prompts, L0(>0.01)={l0}, L1={l1:.3f}",
    ]
    for rank, idx in enumerate(order[:top_n]):
        if np.abs(w[idx]) < 1e-6:
            break
        label = mixture.cache.prompts[idx] if mixture.cache.prompts[idx] else "<empty>"
        lines.append(f"  {rank + 1:2d}. w={w[idx]:+.4f}  \"{label}\"")
    return "\n".join(lines)


def mixture_support(
    mixture: Mixture,
    threshold: float = 0.01,
) -> list[tuple[str, float]]:
    """Active (prompt, weight) pairs sorted by |weight| descending."""
    w = mixture.weights
    active = np.where(np.abs(w) > threshold)[0]
    order = sorted(active, key=lambda i: -abs(w[i]))
    return [(mixture.cache.prompts[i], float(w[i])) for i in order]


@jaxtyped(typechecker=beartype)
def top_k_agreement(
    p: Float[Tensor, "N V"],
    q: Float[Tensor, "N V"],
    k: int = 10,
) -> float:
    """Fraction of top-k tokens shared between two distributions, averaged over N."""
    N = p.shape[0]
    total = 0.0
    p_topk = torch.topk(p, k, dim=-1).indices  # [N, k]
    q_topk = torch.topk(q, k, dim=-1).indices  # [N, k]
    for n in range(N):
        p_set = set(p_topk[n].tolist())
        q_set = set(q_topk[n].tolist())
        total += len(p_set & q_set) / k
    return total / N


def support_recovery(
    true_weights: Float[np.ndarray, " K"],
    recovered_weights: Float[np.ndarray, " K"],
    threshold: float = 0.01,
) -> SupportMetrics:
    """Precision/recall/F1 for identifying active prompt indices.

    Args:
        true_weights: Ground-truth weight vector.
        recovered_weights: Recovered weight vector.
        threshold: Minimum ``|w|`` to count a prompt as active.

    Returns:
        A :class:`SupportMetrics` dataclass with fields:
        ``precision``, ``recall``, ``f1``, ``true_size``, ``recovered_size``.
    """
    true_active = set(np.where(np.abs(true_weights) > threshold)[0])
    recovered_active = set(np.where(np.abs(recovered_weights) > threshold)[0])

    if len(recovered_active) == 0 and len(true_active) == 0:
        return SupportMetrics(1.0, 1.0, 1.0, 0, 0)

    tp = len(true_active & recovered_active)
    precision = tp / max(len(recovered_active), 1)
    recall = tp / max(len(true_active), 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return SupportMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_size=len(true_active),
        recovered_size=len(recovered_active),
    )


@jaxtyped(typechecker=beartype)
def condition_number(
    cache: LogprobCache,
    max_vocab: int = 5000,
) -> DictionaryDiagnostics:
    """SVD-based conditioning analysis of the dictionary matrix A [NV, K].

    Computes singular values, condition number, effective rank
    (Shannon entropy of normalized squared singular values), and
    mean pairwise prompt cosine similarity.
    """
    lp = cache.logprobs
    K, N, V = lp.shape

    # Optionally cap vocabulary for memory
    if V > max_vocab:
        var = lp.var(dim=0).mean(dim=0)
        _, top_idx = torch.topk(var, max_vocab)
        lp = lp[:, :, top_idx.sort().values]
        V = max_vocab

    # Build A matrix: [NV, K]
    A = lp.permute(1, 2, 0).reshape(N * V, K).numpy()

    # SVD
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    cond = float(s.max() / s.min()) if s.min() > 0 else float("inf")

    # Effective rank via Shannon entropy
    s_sq = s**2
    p = s_sq / s_sq.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_rank = float(np.exp(entropy))

    # Mean pairwise cosine similarity
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)  # [1, K]
    A_normed = A / (col_norms + 1e-10)
    cosine_matrix = A_normed.T @ A_normed  # [K, K]
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    mean_cos = float(cosine_matrix[mask].mean())

    return DictionaryDiagnostics(
        singular_values=s,
        condition_number=cond,
        effective_rank=eff_rank,
        mean_cosine=mean_cos,
    )


@jaxtyped(typechecker=beartype)
def pairwise_cosine(cache: LogprobCache) -> Float[Tensor, "K K"]:
    """Cosine similarity between all prompt pairs (averaged over test inputs).

    For each test input n, computes the K x K cosine matrix over the
    vocab dimension, then averages across inputs.
    """
    lp = cache.logprobs  # [K, N, V]
    K, N, V = lp.shape
    # Normalize each prompt's logprob vector per input
    norms = lp.norm(dim=-1, keepdim=True).clamp(min=1e-10)  # [K, N, 1]
    lp_normed = lp / norms  # [K, N, V]
    # Cosine matrix per input, then average
    # cosine[k1, k2, n] = sum_v lp_normed[k1, n, v] * lp_normed[k2, n, v]
    cos_per_input = einsum(
        lp_normed, lp_normed, "K1 N V, K2 N V -> K1 K2 N"
    )
    return cos_per_input.mean(dim=-1)  # [K, K]
