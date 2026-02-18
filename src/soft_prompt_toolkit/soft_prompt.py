"""SoftPrompt module and training utilities.

Provides the :class:`SoftPrompt` ``nn.Module`` for learnable prompt embeddings,
and functions for training soft prompts via:

- **Language modeling loss** on text data (standard soft prompt tuning).
- **KL minimization** against a target log-probability distribution
  (for synthetic recovery experiments).
- **Residual training** to close the gap between a hard-prompt mixture
  and a target distribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ._utils import get_embed_layer
from .evaluate import renormalize
from .types import Mixture


class SoftPrompt(nn.Module):
    """Learnable embedding-space prompt prepended to model inputs.

    Holds a ``[num_tokens, hidden_size]`` parameter tensor that is
    prepended to the model's input embeddings during forward passes.

    Attributes:
        prompt_embeddings: The learnable parameter, shape ``[T, H]``.
    """

    prompt_embeddings: nn.Parameter

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        init_embeddings: Float[Tensor, "T H"] | None = None,
    ) -> None:
        """Initialize the soft prompt.

        Args:
            num_tokens: Number of soft prompt tokens.
            hidden_size: Embedding dimension (must match the model).
            init_embeddings: Optional initialization tensor. If ``None``,
                initializes from ``N(0, 0.02)``.
        """
        super().__init__()
        if init_embeddings is not None:
            self.prompt_embeddings = nn.Parameter(init_embeddings.clone().float())
        else:
            self.prompt_embeddings = nn.Parameter(
                torch.randn(num_tokens, hidden_size) * 0.02
            )

    def forward(self) -> Float[Tensor, "T H"]:
        """Return the prompt embeddings."""
        return self.prompt_embeddings

    @jaxtyped(typechecker=beartype)
    def prepend_to(
        self, input_embeds: Float[Tensor, "B S H"]
    ) -> Float[Tensor, "B Splus H"]:
        """Concatenate prompt embeddings before input embeddings.

        Args:
            input_embeds: Input token embeddings, shape ``[B, S, H]``.

        Returns:
            Concatenated embeddings, shape ``[B, T+S, H]``.
        """
        B = input_embeds.shape[0]
        prompt = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        prompt = prompt.to(dtype=input_embeds.dtype, device=input_embeds.device)
        return torch.cat([prompt, input_embeds], dim=1)


def soft_prompt_from_text(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int | None = None,
) -> SoftPrompt:
    """Create a SoftPrompt initialized from a hard prompt's token embeddings.

    Tokenizes the text, looks up embeddings in the model's embedding table,
    and uses them as initialization. Pads with ``N(0, 0.02)`` or truncates
    to reach ``num_tokens``.

    Args:
        text: Initialization text string.
        model: The model to extract embeddings from.
        tokenizer: Corresponding tokenizer.
        num_tokens: Desired number of tokens. If ``None``, uses the
            number of tokens in the text.

    Returns:
        A new :class:`SoftPrompt` initialized from the text.
    """
    embed_layer = get_embed_layer(model)
    device = next(model.parameters()).device

    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    with torch.no_grad():
        embeds = embed_layer(input_ids).squeeze(0).float()

    n_text = embeds.shape[0]
    hidden_size = embeds.shape[1]

    if num_tokens is None:
        num_tokens = n_text

    if n_text >= num_tokens:
        init = embeds[:num_tokens]
    else:
        pad = torch.randn(num_tokens - n_text, hidden_size, device=device) * 0.02
        init = torch.cat([embeds, pad], dim=0)

    return SoftPrompt(num_tokens, hidden_size, init.cpu())


def train_soft_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_texts: list[str],
    num_tokens: int = 20,
    epochs: int = 15,
    lr: float = 3e-3,
    batch_size: int = 4,
    max_seq_len: int = 64,
    init_text: str | None = None,
    grad_clip: float = 1.0,
    min_tokens: int = 20,
    verbose: bool = True,
) -> tuple[SoftPrompt, list[float]]:
    """Train a soft prompt via language modeling loss on text data.

    Freezes all model parameters and only optimizes the soft prompt
    embeddings using Adam. The loss is cross-entropy on the continuation
    tokens (after the soft prompt).

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        train_texts: Training text strings.
        num_tokens: Number of soft prompt tokens to learn.
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        batch_size: Number of examples per gradient step.
        max_seq_len: Maximum token length per training example.
        init_text: Optional text to initialize the soft prompt from.
        grad_clip: Maximum gradient norm for clipping.
        min_tokens: Minimum token length for training examples. Shorter
            examples are dropped with a warning.
        verbose: Print loss at each epoch.

    Returns:
        A tuple ``(soft_prompt, losses)`` where ``losses`` is a list
        of per-epoch average losses.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = get_embed_layer(model)
    hidden_size = model.config.hidden_size

    # Initialize
    if init_text:
        soft_prompt = soft_prompt_from_text(init_text, model, tokenizer, num_tokens)
    else:
        soft_prompt = SoftPrompt(num_tokens, hidden_size)
    soft_prompt = soft_prompt.to(device)

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(soft_prompt.parameters(), lr=lr)

    # Tokenize training data
    train_data: list[Tensor] = []
    for text in train_texts:
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_seq_len
        ).input_ids
        if len(ids) >= min_tokens:
            train_data.append(torch.tensor(ids[:max_seq_len]))
    n_dropped = len(train_texts) - len(train_data)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped}/{len(train_texts)} training examples "
            f"shorter than {min_tokens} tokens.",
            stacklevel=2,
        )
    if not train_data:
        raise ValueError(
            f"No training examples survived filtering (min {min_tokens} tokens)"
        )

    if verbose:
        print(f"Training soft prompt: {num_tokens} tokens, {epochs} epochs, "
              f"{len(train_data)} examples")

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(train_data))

        for batch_start in range(0, len(train_data), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            n_batch = len(batch_indices)
            batch_loss_sum = 0.0

            for idx in batch_indices:
                input_ids = train_data[idx].to(device)
                with torch.no_grad():
                    input_embeds = embed_layer(input_ids.unsqueeze(0)).to(dtype)

                prompt_embeds = soft_prompt().unsqueeze(0).to(dtype)
                full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

                logits = model(inputs_embeds=full_embeds).logits[0]
                shift_logits = logits[num_tokens:-1]
                shift_labels = input_ids[1:]

                min_len = min(shift_logits.shape[0], shift_labels.shape[0])
                loss = F.cross_entropy(
                    shift_logits[:min_len].float(), shift_labels[:min_len]
                )
                (loss / n_batch).backward()
                batch_loss_sum += loss.item()

            torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += batch_loss_sum / n_batch
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    return soft_prompt, losses


@torch.no_grad()
def evaluate_soft_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    soft_prompt: SoftPrompt,
    texts: list[str],
    max_seq_len: int = 64,
    min_tokens: int = 20,
) -> float:
    """Compute mean cross-entropy loss on texts with a soft prompt prepended.

    Mirrors the training loss of :func:`train_soft_prompt` but on held-out
    data, giving a quantitative measure of soft prompt quality.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        soft_prompt: Trained soft prompt module.
        texts: Evaluation text strings.
        max_seq_len: Maximum token length per example.
        min_tokens: Minimum token length; shorter examples are skipped.

    Returns:
        Mean cross-entropy loss across all examples.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = get_embed_layer(model)
    num_tokens = soft_prompt.prompt_embeddings.shape[0]

    eval_data: list[Tensor] = []
    for text in texts:
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_seq_len
        ).input_ids
        if len(ids) >= min_tokens:
            eval_data.append(torch.tensor(ids[:max_seq_len]))

    if not eval_data:
        raise ValueError(
            f"No evaluation examples survived filtering (min {min_tokens} tokens)"
        )

    total_loss = 0.0
    for ids_tensor in eval_data:
        input_ids = ids_tensor.to(device)
        input_embeds = embed_layer(input_ids.unsqueeze(0)).to(dtype)

        prompt_embeds = soft_prompt().unsqueeze(0).to(dtype=dtype, device=device)
        full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        logits = model(inputs_embeds=full_embeds).logits[0]
        shift_logits = logits[num_tokens:-1]
        shift_labels = input_ids[1:]

        min_len = min(shift_logits.shape[0], shift_labels.shape[0])
        loss = F.cross_entropy(
            shift_logits[:min_len].float(), shift_labels[:min_len]
        )
        total_loss += loss.item()

    return total_loss / len(eval_data)


def train_soft_prompt_to_distribution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    target_logprobs: Float[Tensor, "N V"],
    test_inputs: list[str],
    shared_indices: Int[Tensor, " V"],
    num_tokens: int = 20,
    epochs: int = 30,
    lr: float = 3e-3,
    batch_size: int = 10,
    device: str = "cuda",
    verbose: bool = True,
) -> SoftPrompt:
    """Train a soft prompt to match a target distribution via KL minimization.

    Minimizes ``KL(target || soft_prompt_prediction)`` on the shared
    vocabulary for each test input. Used in synthetic recovery experiments.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        target_logprobs: Target logprobs on shared vocab, shape ``[N, V]``.
        test_inputs: List of N test input prefix strings.
        shared_indices: Vocabulary indices for the shared set.
        num_tokens: Number of soft prompt tokens.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Number of test inputs per gradient step.
        device: Device for training.
        verbose: Print progress every 5 epochs.

    Returns:
        The trained :class:`SoftPrompt` (best-epoch weights restored).
    """
    dtype = next(model.parameters()).dtype
    embed_layer = get_embed_layer(model)
    hidden_size = embed_layer.weight.shape[1]

    model.gradient_checkpointing_enable()

    soft_prompt = SoftPrompt(num_tokens, hidden_size).to(device).to(dtype)

    # Random embedding initialization
    rng = np.random.RandomState(123)
    init_ids = rng.choice(tokenizer.vocab_size, size=num_tokens, replace=False)
    with torch.no_grad():
        init_embeds = embed_layer(torch.tensor(init_ids, device=device))
        soft_prompt.prompt_embeddings.data = init_embeds.to(dtype)

    optimizer = torch.optim.Adam(soft_prompt.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Precompute tokenized inputs
    all_input_ids = [
        tokenizer(inp, return_tensors="pt", add_special_tokens=True).input_ids.to(
            device
        )
        for inp in test_inputs
    ]

    target_probs = torch.exp(target_logprobs).to(device)
    target_lp_dev = target_logprobs.to(device)
    shared_dev = shared_indices.to(device)
    N = len(test_inputs)

    best_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        total_loss = 0.0
        perm = torch.randperm(N)

        for start in range(0, N, batch_size):
            batch_idx = perm[start : start + batch_size]
            n_batch = len(batch_idx)
            optimizer.zero_grad()

            for idx in batch_idx:
                ii = idx.item()
                input_embeds = embed_layer(all_input_ids[ii])
                full_embeds = soft_prompt.prepend_to(input_embeds)
                logits = model(inputs_embeds=full_embeds).logits[0, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)

                log_probs_shared = log_probs[shared_dev]
                kl = (target_probs[ii] * (target_lp_dev[ii] - log_probs_shared)).sum()
                (kl / n_batch).backward()
                total_loss += kl.item() / n_batch

            optimizer.step()

        scheduler.step()
        n_batches = (N + batch_size - 1) // batch_size
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in soft_prompt.state_dict().items()}

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"  Epoch {epoch + 1}/{epochs}: "
                f"avg KL = {avg_loss:.4f} (best={best_loss:.4f})"
            )

    if best_state is not None:
        soft_prompt.load_state_dict(best_state)

    model.gradient_checkpointing_disable()
    return soft_prompt


def train_residual(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    current_mixture: Mixture,
    target_logprobs: Float[Tensor, "N V"],
    test_inputs: list[str],
    shared_indices: Int[Tensor, " V"],
    num_tokens: int = 10,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 10,
    learn_weight: bool = True,
    init_weight: float = 1.0,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[SoftPrompt, float, list[float]]:
    """Train a soft prompt to close the gap between a mixture and a target.

    The augmented model is::

        log p(t|x) = sum_i w_i log p_i(t|x) + w_new log p_soft(t|x) - log Z(x)

    Mixture weights are frozen. Optimizes the soft prompt embeddings and
    (optionally) the scalar ``w_new``.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        current_mixture: The existing hard-prompt mixture (weights frozen).
        target_logprobs: Target logprobs, shape ``[N, V]``.
        test_inputs: List of N test input prefix strings.
        shared_indices: Shared vocabulary indices.
        num_tokens: Number of soft prompt tokens (typically smaller
            than the original, since it only captures the residual).
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Test inputs per gradient step.
        learn_weight: If ``True``, jointly optimize ``w_new``.
        init_weight: Initial value for ``w_new``.
        device: Device for training.
        verbose: Print progress.

    Returns:
        A tuple ``(residual_soft_prompt, learned_weight, per_epoch_losses)``.
    """
    dtype = next(model.parameters()).dtype
    embed_layer = get_embed_layer(model)
    hidden_size = embed_layer.weight.shape[1]

    soft_prompt = SoftPrompt(num_tokens, hidden_size).to(device)

    w_new = torch.tensor(
        init_weight, dtype=torch.float32, device=device, requires_grad=learn_weight
    )

    # Precompute frozen mixture logprobs (unnormalized)
    w_mix = torch.as_tensor(current_mixture.weights, dtype=torch.float32)
    mixture_lp_unnorm = einsum(
        w_mix, current_mixture.cache.logprobs, "K, K N V -> N V"
    ).to(device)

    # Precompute target
    target_lp_norm = renormalize(target_logprobs).to(device)
    target_p = target_lp_norm.exp()

    # Tokenize inputs
    all_input_ids = [
        tokenizer(inp, return_tensors="pt", add_special_tokens=True).input_ids.to(
            device
        )
        for inp in test_inputs
    ]
    shared_dev = shared_indices.to(device)

    params: list[torch.nn.Parameter | Tensor] = list(soft_prompt.parameters())
    if learn_weight:
        params.append(w_new)
    optimizer = torch.optim.Adam(params, lr=lr)

    N = len(test_inputs)
    losses: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_kl = torch.tensor(0.0, device=device)

            for idx in batch_idx:
                n = idx.item()
                input_embeds = embed_layer(all_input_ids[n])
                prompt_embeds = soft_prompt().unsqueeze(0).to(dtype)
                full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
                logits = model(inputs_embeds=full_embeds).logits[0, -1, :]
                soft_lp = F.log_softmax(logits.float(), dim=-1)[shared_dev]

                augmented_unnorm = mixture_lp_unnorm[n] + w_new * soft_lp
                augmented_lp = augmented_unnorm - torch.logsumexp(
                    augmented_unnorm, dim=-1
                )

                kl = (target_p[n] * (target_lp_norm[n] - augmented_lp)).sum()
                batch_kl = batch_kl + kl

            batch_kl = batch_kl / len(batch_idx)
            optimizer.zero_grad()
            batch_kl.backward()
            torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_kl.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"  Epoch {epoch + 1}/{epochs}: KL = {avg_loss:.4f}, "
                f"w_new = {w_new.item():.4f}"
            )

    return soft_prompt, w_new.item(), losses


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    soft_prompt: SoftPrompt,
    prefix: str = "",
    max_new_tokens: int = 50,
    num_samples: int = 5,
    temperature: float = 0.8,
) -> list[str]:
    """Generate text samples with a soft prompt prepended.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        soft_prompt: Trained soft prompt module.
        prefix: Optional text prefix after the soft prompt.
        max_new_tokens: Maximum tokens to generate per sample.
        num_samples: Number of samples to generate.
        temperature: Sampling temperature. Use ``0`` for greedy (argmax)
            decoding.

    Returns:
        List of generated text strings.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = get_embed_layer(model)
    samples: list[str] = []

    for _ in range(num_samples):
        prompt_embeds = soft_prompt().unsqueeze(0).to(dtype=dtype, device=device)

        if prefix:
            prefix_ids = tokenizer(
                prefix, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            prefix_embeds = embed_layer(prefix_ids).to(dtype)
            current_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)
            generated_ids = prefix_ids[0].tolist()
        else:
            current_embeds = prompt_embeds
            generated_ids = []

        for _ in range(max_new_tokens):
            logits = model(inputs_embeds=current_embeds).logits[0, -1, :]
            if temperature <= 0:
                next_token = logits.float().argmax().item()
            else:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            next_embed = embed_layer(
                torch.tensor([[next_token]], device=device)
            ).to(dtype)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        samples.append(text)

    return samples
