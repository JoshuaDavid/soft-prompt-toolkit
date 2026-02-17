"""soft_prompt_toolkit: Decompose soft prompts into weighted hard prompt mixtures.

Implements the Product-of-Experts decomposition model::

    log p_combined(t|x) = sum_i w_i * log p_i(t|x) - log Z(x)

where ``w_i`` are sparse weights over a dictionary of hard prompts,
recovered via L1-penalized least-squares optimization.
"""

from .types import (
    DictionaryDiagnostics,
    GreedyStep,
    LogprobCache,
    Mixture,
    SupportMetrics,
)
from .collect import (
    cap_vocab,
    collect_dictionary,
    collect_target_hard,
    collect_target_soft,
    load_cache,
    make_synthetic_target,
    sample_cache,
    save_cache,
    subset_cache,
)
from .solve import find_weights, greedy_select
from .evaluate import (
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
from .soft_prompt import (
    SoftPrompt,
    generate,
    soft_prompt_from_text,
    train_residual,
    train_soft_prompt,
    train_soft_prompt_to_distribution,
)

__all__ = [
    # Types
    "LogprobCache",
    "Mixture",
    "DictionaryDiagnostics",
    "SupportMetrics",
    "GreedyStep",
    # Collection
    "collect_dictionary",
    "collect_target_soft",
    "collect_target_hard",
    "cap_vocab",
    "make_synthetic_target",
    "subset_cache",
    "sample_cache",
    "save_cache",
    "load_cache",
    # Solving
    "find_weights",
    "greedy_select",
    # Evaluation
    "mixture_logprobs",
    "mixture_kl",
    "mixture_mean_kl",
    "mixture_summary",
    "mixture_support",
    "renormalize",
    "kl_divergence",
    "top_k_agreement",
    "support_recovery",
    "condition_number",
    "pairwise_cosine",
    # Soft prompt
    "SoftPrompt",
    "soft_prompt_from_text",
    "train_soft_prompt",
    "train_soft_prompt_to_distribution",
    "train_residual",
    "generate",
]
