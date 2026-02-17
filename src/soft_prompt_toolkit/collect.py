"""Log-probability collection from HuggingFace causal language models.

All functions that touch models and produce :class:`LogprobCache` or
target tensors live here.

The standard workflow is a two-pass approach:

1. **Pass 1 (vocab discovery)**: For each ``(prompt, input)`` pair, extract
   the top-k token indices. Their union forms the shared vocabulary.
2. **Pass 2 (collection)**: Re-run each pair and collect ``log_softmax``
   values restricted to the shared vocabulary.

This keeps memory tractable while preserving the tokens that matter most
for distinguishing between prompts.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .types import LogprobCache


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def collect_dictionary(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    test_inputs: list[str],
    top_k: int = 1000,
    device: str = "cuda",
    verbose: bool = True,
) -> LogprobCache:
    """Two-pass logprob collection for a list of hard prompts.

    Args:
        model: A HuggingFace causal LM in eval mode.
        tokenizer: Corresponding tokenizer.
        prompts: List of K dictionary prompt strings.
        test_inputs: List of N test input prefix strings.
        top_k: Number of top tokens per ``(prompt, input)`` pair for
            building the shared vocabulary.
        device: Device for model inputs.
        verbose: Print progress updates.

    Returns:
        A :class:`LogprobCache` with logprobs of shape ``[K, N, V]``.
    """
    K = len(prompts)
    N = len(test_inputs)

    # Pass 1: discover shared vocabulary
    if verbose:
        print("Pass 1: Discovering shared vocabulary...")
    all_indices: set[int] = set()
    total = K * N
    done = 0
    for p in prompts:
        for inp in test_inputs:
            text = p + inp
            inputs = tokenizer(text, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0, -1, :]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            top_idxs = torch.topk(logprobs, top_k).indices
            all_indices.update(top_idxs.cpu().tolist())
            done += 1
            if verbose and done % 100 == 0:
                print(f"  Pass 1: {done}/{total}")

    shared_indices = torch.tensor(sorted(all_indices), dtype=torch.long)
    V = len(shared_indices)
    if verbose:
        print(f"  Shared vocab size: {V}")

    # Pass 2: collect logprobs on shared vocab
    if verbose:
        print("Pass 2: Collecting logprobs...")
    shared_gpu = shared_indices.to(device)
    dict_logprobs = torch.zeros(K, N, V)

    done = 0
    for pi, p in enumerate(prompts):
        for ii, inp in enumerate(test_inputs):
            text = p + inp
            inputs = tokenizer(text, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0, -1, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
            dict_logprobs[pi, ii, :] = lp[shared_gpu].cpu()
            done += 1
            if verbose and done % 100 == 0:
                print(f"  Pass 2: {done}/{total}")

    return LogprobCache(
        prompts=prompts,
        test_inputs=test_inputs,
        shared_indices=shared_indices,
        logprobs=dict_logprobs,
    )


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def collect_target_soft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    soft_prompt: torch.nn.Module,
    test_inputs: list[str],
    shared_indices: Int[Tensor, " V"],
    device: str = "cuda",
) -> Float[Tensor, "N V"]:
    """Collect logprobs from a soft-prompted model on a shared vocab.

    Prepends ``soft_prompt`` embeddings to each test input's token embeddings,
    runs a forward pass, and extracts ``log_softmax`` at the final position,
    restricted to ``shared_indices``.

    Args:
        model: A HuggingFace causal LM in eval mode.
        tokenizer: Corresponding tokenizer.
        soft_prompt: Module whose ``forward()`` returns embeddings of shape
            ``[num_tokens, hidden_size]``.
        test_inputs: List of N test input prefix strings.
        shared_indices: Vocabulary token IDs in the shared set.
        device: Device for model inputs.

    Returns:
        Float tensor of shape ``[N, V]``.
    """
    embed_layer = model.model.embed_tokens
    dtype = next(model.parameters()).dtype
    N = len(test_inputs)
    V = len(shared_indices)
    shared_gpu = shared_indices.to(device)
    target_lp = torch.zeros(N, V)

    for ii, inp in enumerate(test_inputs):
        prompt_embeds = soft_prompt().unsqueeze(0).to(dtype=dtype, device=device)
        input_ids = tokenizer(
            inp, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        input_embeds = embed_layer(input_ids).to(dtype)
        full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        logits = model(inputs_embeds=full_embeds).logits[0, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        target_lp[ii] = lp[shared_gpu].cpu()

    return target_lp


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def collect_target_hard(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    test_inputs: list[str],
    shared_indices: Int[Tensor, " V"],
    device: str = "cuda",
) -> Float[Tensor, "N V"]:
    """Collect logprobs for a single hard prompt on a shared vocab.

    Args:
        model: A HuggingFace causal LM in eval mode.
        tokenizer: Corresponding tokenizer.
        prompt: The hard prompt string.
        test_inputs: List of N test input prefix strings.
        shared_indices: Vocabulary token IDs in the shared set.
        device: Device for model inputs.

    Returns:
        Float tensor of shape ``[N, V]``.
    """
    N = len(test_inputs)
    V = len(shared_indices)
    shared_gpu = shared_indices.to(device)
    target_lp = torch.zeros(N, V)

    for ii, inp in enumerate(test_inputs):
        text = prompt + inp
        inputs = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0, -1, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        target_lp[ii] = lp[shared_gpu].cpu()

    return target_lp


@jaxtyped(typechecker=beartype)
def cap_vocab(
    dict_logprobs: Float[Tensor, "K N V"],
    max_vocab: int,
    target_logprobs: Float[Tensor, "N V"] | None = None,
    target_weight: float = 0.5,
) -> Int[Tensor, " Vprime"]:
    """Select the most informative tokens by cross-prompt variance.

    Ranks tokens by a score combining variance across dictionary prompts
    (discriminating power) and, optionally, probability mass under the
    target distribution.

    Args:
        dict_logprobs: Dictionary logprobs, shape ``[K, N, V]``.
        max_vocab: Maximum number of tokens to retain.
        target_logprobs: Optional target logprobs, shape ``[N, V]``.
            If provided, tokens with high target probability get upweighted.
        target_weight: Relative weight of the target probability term.

    Returns:
        Sorted int64 tensor of selected vocabulary indices, shape ``[V']``.
    """
    V = dict_logprobs.shape[2]
    if V <= max_vocab:
        return torch.arange(V, dtype=torch.long)

    # Variance across prompts (averaged over inputs)
    var_across_prompts = dict_logprobs.var(dim=0).mean(dim=0)  # [V]
    var_norm = var_across_prompts / (var_across_prompts.max() + 1e-10)

    if target_logprobs is not None:
        target_prob_mass = torch.exp(target_logprobs).mean(dim=0)  # [V]
        prob_norm = target_prob_mass / (target_prob_mass.max() + 1e-10)
        score = var_norm + target_weight * prob_norm
    else:
        score = var_norm

    _, top_indices = torch.topk(score, max_vocab)
    return top_indices.sort().values


@jaxtyped(typechecker=beartype)
def make_synthetic_target(
    cache: LogprobCache,
    weights: Float[np.ndarray, " K"] | dict[int, float],
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> Float[Tensor, "N V"]:
    """Create PoE target logprobs from known weights.

    Computes the weighted combination of dictionary logprobs, renormalizes,
    and optionally adds Gaussian noise. Useful for synthetic recovery
    experiments.

    Args:
        cache: Dictionary logprob data.
        weights: Weight vector (length K) or dict mapping prompt index to weight.
        noise_sigma: Standard deviation of Gaussian noise to add.
        seed: Random seed for noise generation.

    Returns:
        Float tensor of shape ``[N, V]`` with (possibly noisy) target logprobs.
    """
    K = cache.logprobs.shape[0]

    if isinstance(weights, dict):
        w_arr = np.zeros(K)
        for idx, val in weights.items():
            w_arr[idx] = val
        weights = w_arr

    w = torch.as_tensor(weights, dtype=torch.float32)
    combined = einsum(w, cache.logprobs, "K, K N V -> N V")
    target = combined - torch.logsumexp(combined, dim=-1, keepdim=True)

    if noise_sigma > 0:
        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn_like(target, generator=rng) * noise_sigma
        target = target + noise
        target = target - torch.logsumexp(target, dim=-1, keepdim=True)

    return target


def subset_cache(cache: LogprobCache, indices: list[int]) -> LogprobCache:
    """Subset a LogprobCache to specific prompt indices.

    Args:
        cache: The original cache.
        indices: List of prompt indices to keep.

    Returns:
        A new :class:`LogprobCache` with only the selected prompts.
    """
    return LogprobCache(
        prompts=[cache.prompts[i] for i in indices],
        test_inputs=cache.test_inputs,
        shared_indices=cache.shared_indices,
        logprobs=cache.logprobs[indices],
    )


def sample_cache(cache: LogprobCache, n: int, seed: int = 42) -> LogprobCache:
    """Randomly sample n prompts from a LogprobCache.

    Args:
        cache: The original cache.
        n: Number of prompts to sample.
        seed: Random seed for reproducibility.

    Returns:
        A new :class:`LogprobCache` with n randomly selected prompts.
    """
    rng = torch.Generator().manual_seed(seed)
    K = cache.logprobs.shape[0]
    perm = torch.randperm(K, generator=rng)[:n].sort().values.tolist()
    return subset_cache(cache, perm)


def save_cache(cache: LogprobCache, path: str) -> None:
    """Save a LogprobCache to disk via ``torch.save``.

    Args:
        cache: The cache to save.
        path: File path to write to.
    """
    torch.save(
        {
            "prompts": cache.prompts,
            "test_inputs": cache.test_inputs,
            "shared_indices": cache.shared_indices,
            "logprobs": cache.logprobs,
        },
        path,
    )


def load_cache(path: str) -> LogprobCache:
    """Load a LogprobCache from disk.

    Args:
        path: File path to load from.

    Returns:
        The loaded :class:`LogprobCache`.
    """
    data = torch.load(path, weights_only=False)
    return LogprobCache(
        prompts=data["prompts"],
        test_inputs=data["test_inputs"],
        shared_indices=data["shared_indices"],
        logprobs=data["logprobs"],
    )
