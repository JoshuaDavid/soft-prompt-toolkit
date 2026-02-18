# soft-prompt-toolkit

Decompose soft prompts into weighted mixtures of hard prompts via Product-of-Experts.

Given a soft prompt and a dictionary of hard prompts, this toolkit recovers sparse
weights `w` such that:

```
log p_combined(t|x) = sum_i w_i * log p_i(t|x) - log Z(x)
```

approximates the soft prompt's next-token distribution. The weights are found by
solving an L1-penalized least-squares problem over cached log-probabilities.

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quickstart

### 1. Collect log-probabilities

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from soft_prompt_toolkit import collect_dictionary, collect_target_hard

model = AutoModelForCausalLM.from_pretrained("your-model").eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Dictionary of candidate hard prompts
prompts = ["Translate to French:", "Summarize:", "Answer the question:", ...]

# Shared test inputs
test_inputs = ["The cat sat on", "Once upon a time", ...]

# Two-pass collection: discover shared vocab, then collect logprobs
cache = collect_dictionary(model, tokenizer, prompts, test_inputs, top_k=1000)

# Collect target (the prompt you want to decompose)
target = collect_target_hard(model, tokenizer, "Your target prompt",
                             test_inputs, cache.shared_indices)
```

For soft prompt targets, use `collect_target_soft` instead.

### 2. (Optional) Cap vocabulary for speed

```python
from soft_prompt_toolkit import cap_vocab

indices = cap_vocab(cache.logprobs, max_vocab=5000, target_logprobs=target)
cache_small = LogprobCache(
    prompts=cache.prompts, test_inputs=cache.test_inputs,
    shared_indices=cache.shared_indices[indices],
    logprobs=cache.logprobs[:, :, indices],
)
target_small = target[:, indices]
```

### 3. Solve for weights

```python
from soft_prompt_toolkit import find_weights

mixture = find_weights(cache, target, alpha=0.01)
# mixture.weights is a sparse numpy array of length K
# mixture.cache is the LogprobCache
```

The solver auto-selects GPU (hybrid `A^T A` precomputation on CUDA, L-BFGS-B on
CPU) or pure CPU. Pass `device="cuda"` explicitly for large problems.

### 4. Evaluate

```python
from soft_prompt_toolkit import (
    mixture_mean_kl, mixture_summary, mixture_support, mixture_logprobs,
)

print(f"Mean KL: {mixture_mean_kl(mixture, target):.4f}")
print(mixture_summary(mixture))

# Active prompts sorted by |weight|
for prompt, weight in mixture_support(mixture):
    print(f"  w={weight:+.4f}  {prompt}")
```

### 5. Greedy forward selection

```python
from soft_prompt_toolkit import greedy_select

steps = greedy_select(cache, target, alpha=0.01, max_steps=10)
for s in steps:
    print(f"Step {s.step}: KL={s.kl:.4f}  +\"{s.prompt[:50]}\"")
```

## Soft prompt training

```python
from soft_prompt_toolkit import train_soft_prompt, evaluate_soft_prompt, generate

# Train
sp, losses = train_soft_prompt(model, tokenizer, train_texts, num_tokens=20, epochs=15)

# Evaluate on held-out data
eval_loss = evaluate_soft_prompt(model, tokenizer, sp, test_texts)

# Generate
samples = generate(model, tokenizer, sp, prefix="Once", temperature=0.8)
```

### Residual training

Close the gap between a hard-prompt mixture and a target distribution by training
a small residual soft prompt:

```python
from soft_prompt_toolkit import train_residual

residual_sp, w_new, losses = train_residual(
    model, tokenizer, mixture, target_logprobs,
    test_inputs, shared_indices, num_tokens=10,
)
```

## Diagnostics

```python
from soft_prompt_toolkit import condition_number, pairwise_cosine

diag = condition_number(cache)
print(f"Condition number: {diag.condition_number:.1f}")
print(f"Effective rank: {diag.effective_rank:.1f}")
print(f"Mean cosine: {diag.mean_cosine:.3f}")

cosine_matrix = pairwise_cosine(cache)  # [K, K] tensor
```

## Saving and loading

```python
from soft_prompt_toolkit import save_cache, load_cache

save_cache(cache, "my_cache.pt")
cache = load_cache("my_cache.pt")
```

## API reference

### Data types

| Type | Fields |
|------|--------|
| `LogprobCache` | `prompts`, `test_inputs`, `shared_indices`, `logprobs [K,N,V]` |
| `Mixture` | `cache`, `weights [K]` |
| `SupportMetrics` | `precision`, `recall`, `f1`, `true_size`, `recovered_size` |
| `DictionaryDiagnostics` | `singular_values`, `condition_number`, `effective_rank`, `mean_cosine` |
| `GreedyStep` | `step`, `prompt_idx`, `prompt`, `kl`, `weights` |

### Functions

**Collection** (`collect.py`): `collect_dictionary`, `collect_target_soft`, `collect_target_hard`, `cap_vocab`, `make_synthetic_target`, `subset_cache`, `sample_cache`, `save_cache`, `load_cache`

**Solving** (`solve.py`): `find_weights`, `greedy_select`

**Evaluation** (`evaluate.py`): `mixture_logprobs`, `mixture_kl`, `mixture_mean_kl`, `mixture_summary`, `mixture_support`, `renormalize`, `kl_divergence`, `top_k_agreement`, `support_recovery`, `condition_number`, `pairwise_cosine`

**Soft prompts** (`soft_prompt.py`): `SoftPrompt`, `soft_prompt_from_text`, `train_soft_prompt`, `evaluate_soft_prompt`, `train_soft_prompt_to_distribution`, `train_residual`, `generate`

**Utilities**: `get_embed_layer`
