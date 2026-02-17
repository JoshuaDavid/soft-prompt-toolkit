"""Weight solvers for the Product-of-Experts decomposition.

Solves the L1-penalized least-squares problem::

    min_w  0.5 * mean(||A w - b||^2)  +  alpha * ||w||_1

where ``A`` is the ``[NV, K]`` matrix of flattened dictionary logprobs
and ``b`` is the ``[NV]`` vector of flattened target logprobs.

Two solver backends:
  - **GPU hybrid**: precompute ``A^T A`` ``[K, K]`` and ``A^T b`` ``[K]``
    on GPU, then run scipy L-BFGS-B on CPU with the compact normal equations.
    Achieves 44--194x speedup for large dictionaries.
  - **CPU**: scipy L-BFGS-B on the full ``[NV, K]`` matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import torch
from einops import rearrange
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor

from .types import GreedyStep, LogprobCache, Mixture


def _solve_gpu(
    dict_logprobs: Tensor,
    target_logprobs: Tensor,
    alpha: float,
    device: str,
    max_iter: int,
) -> np.ndarray:
    """GPU hybrid solver: precompute AtA on CUDA, L-BFGS-B on CPU."""
    K, N, V = dict_logprobs.shape
    m = N * V

    A_gpu = rearrange(dict_logprobs, "K N V -> (N V) K").to(
        device=device, dtype=torch.float32
    )
    b_gpu = rearrange(target_logprobs, "N V -> (N V)").to(
        device=device, dtype=torch.float32
    )

    AtA_gpu = A_gpu.T @ A_gpu  # [K, K]
    Atb_gpu = A_gpu.T @ b_gpu  # [K]

    AtA = AtA_gpu.cpu().numpy().astype(np.float64)
    Atb = Atb_gpu.cpu().numpy().astype(np.float64)

    del A_gpu, b_gpu, AtA_gpu, Atb_gpu
    torch.cuda.empty_cache()

    try:
        w0 = np.linalg.solve(AtA + 1e-6 * np.eye(K), Atb)
    except np.linalg.LinAlgError:
        w0 = np.zeros(K)

    def objective(w: np.ndarray) -> float:
        r_sq = w @ AtA @ w - 2 * Atb @ w
        return 0.5 * r_sq / m + alpha * np.sum(np.abs(w))

    def gradient(w: np.ndarray) -> np.ndarray:
        return (AtA @ w - Atb) / m + alpha * np.sign(w)

    result = scipy.optimize.minimize(
        objective,
        w0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x


def _solve_cpu(
    dict_logprobs: Tensor,
    target_logprobs: Tensor,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    """CPU solver: L-BFGS-B on the full [NV, K] matrix."""
    K, N, V = dict_logprobs.shape
    A = rearrange(dict_logprobs, "K N V -> (N V) K").numpy()
    b = rearrange(target_logprobs, "N V -> (N V)").numpy()
    m = len(b)

    def objective(w: np.ndarray) -> float:
        r = A @ w - b
        return 0.5 * np.mean(r**2) + alpha * np.sum(np.abs(w))

    def gradient(w: np.ndarray) -> np.ndarray:
        r = A @ w - b
        return A.T @ r / m + alpha * np.sign(w)

    w0 = np.linalg.lstsq(A, b, rcond=None)[0]

    result = scipy.optimize.minimize(
        objective,
        w0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x


@jaxtyped(typechecker=beartype)
def find_weights(
    cache: LogprobCache,
    target: Float[Tensor, "N V"],
    alpha: float = 0.01,
    device: str | None = None,
    max_iter: int = 5000,
) -> Mixture:
    """Solve for sparse PoE weights via L1-penalized least squares.

    Solves::

        min_w  0.5 * mean(||A w - b||^2)  +  alpha * ||w||_1

    Auto-selects GPU (precompute ``A^T A`` on CUDA, solve on CPU)
    or pure CPU depending on ``device``.

    Args:
        cache: Dictionary logprob data.
        target: Target logprobs, shape ``[N, V]``.
        alpha: L1 regularization strength. Higher = sparser weights.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        max_iter: Maximum L-BFGS-B iterations.

    Returns:
        A Mixture holding the cache and recovered weight vector.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        w = _solve_gpu(cache.logprobs, target, alpha, device, max_iter)
    else:
        w = _solve_cpu(cache.logprobs, target, alpha, max_iter)

    return Mixture(cache=cache, weights=w)


@jaxtyped(typechecker=beartype)
def greedy_select(
    cache: LogprobCache,
    target: Float[Tensor, "N V"],
    alpha: float = 0.01,
    max_steps: int | None = None,
    early_stop_rtol: float = 0.001,
    verbose: bool = True,
) -> list[GreedyStep]:
    """Greedy forward selection: at each step add the prompt that most reduces KL.

    At each step, evaluates every remaining candidate, adds the one whose
    inclusion most reduces KL divergence, and re-solves for weights.

    Args:
        cache: Dictionary logprob data.
        target: Target logprobs, shape ``[N, V]``.
        alpha: L1 regularization strength for the inner solver.
        max_steps: Maximum prompts to select. ``None`` = select all K.
        early_stop_rtol: Stop if relative KL improvement drops below this.
        verbose: Print progress at each step.

    Returns:
        List of :class:`GreedyStep`, one per selection step.
    """
    from .evaluate import mixture_mean_kl

    K = cache.logprobs.shape[0]
    if max_steps is None:
        max_steps = K

    remaining = set(range(K))
    selected: list[int] = []
    steps: list[GreedyStep] = []

    for step_num in range(1, max_steps + 1):
        best_kl = float("inf")
        best_idx = -1
        best_weights: np.ndarray | None = None

        for candidate in remaining:
            trial = selected + [candidate]
            trial_lp = cache.logprobs[trial]
            trial_cache = LogprobCache(
                prompts=[cache.prompts[i] for i in trial],
                test_inputs=cache.test_inputs,
                shared_indices=cache.shared_indices,
                logprobs=trial_lp,
            )
            mixture = find_weights(trial_cache, target, alpha=alpha, device="cpu")
            kl = mixture_mean_kl(mixture, target)

            if kl < best_kl:
                best_kl = kl
                best_idx = candidate
                best_weights = mixture.weights

        selected.append(best_idx)
        remaining.discard(best_idx)

        step = GreedyStep(
            step=step_num,
            prompt_idx=best_idx,
            prompt=cache.prompts[best_idx],
            kl=best_kl,
            weights=best_weights,
        )
        steps.append(step)

        if verbose:
            print(
                f"  Step {step_num}: KL={best_kl:.4f}  "
                f"+\"{cache.prompts[best_idx][:50]}\""
            )

        # Early stopping
        if len(steps) >= 2:
            prev_kl = steps[-2].kl
            if prev_kl > 0 and (prev_kl - best_kl) / prev_kl < early_stop_rtol:
                if verbose:
                    print(f"  Early stop: relative improvement < {early_stop_rtol}")
                break

        if not remaining:
            break

    return steps
