# Pain Points

## 1. `make_synthetic_target` rejects `np.int64` / `np.float64` in dict keys/values

**Encountered**: When passing a dict with numpy integer keys (from `np.random.choice`) to
`make_synthetic_target(cache, weights={np.int64(0): np.float64(1.0)})`, beartype validation
rejects the call because `np.int64` is not `int` and `np.float64` is not `float`.

**Workaround**: Manually cast to Python builtins:
```python
indices = [int(x) for x in rng.choice(K, size=10, replace=False)]
weights = [float(x) for x in rng.randn(10) * 0.2]
```

**Suggested fix**: Either relax the type annotation from `dict[int, float]` to
`dict[int | np.integer, float | np.floating]`, or add an internal cast at the top
of the function.

## 2. `support_recovery` returns a dataclass but no documentation indicates this

**Encountered**: The agent-generated code used `sr['precision']` (dict-style access)
on the return value of `support_recovery()`, which is actually a `SupportMetrics`
dataclass requiring `sr.precision` (attribute access).

**Workaround**: Use attribute access (`sr.precision`, `sr.recall`, `sr.f1`, `sr.true_size`, `sr.recovered_size`).

**Suggested fix**: Either make `SupportMetrics` a dict (or dict-like via `__getitem__`),
or add prominent documentation that the return is a dataclass with attribute access.

## 3. `SupportMetrics` field names are not discoverable

**Encountered**: Code guessed the field name `l0_recovered` for the number of
recovered active prompts. The actual field is `recovered_size`. There is also
`true_size` for the number of truly active prompts. These names are not obvious
from the function signature alone.

**Workaround**: Read the source of `SupportMetrics` in `types.py` to find the
correct field names: `precision`, `recall`, `f1`, `true_size`, `recovered_size`.

**Suggested fix**: Add field names to the `support_recovery` docstring, or use
more conventional names like `n_true` / `n_recovered` or `l0_true` / `l0_recovered`.

## 4. `find_weights` is extremely slow on CPU with large vocabularies

**Encountered**: Notebook 03 (cross-validation with V=25,337, K=100) timed out
at 600 seconds on CPU. The solver ran 3 alphas × 5 folds × 2 passes = 30 solver
calls, each on a `[50×25337, 100]` matrix. Switching to `device="cuda"` reduced
total runtime to under 60 seconds.

**Workaround**: Always pass `device="cuda"` when a GPU is available. The solver's
`device=None` auto-detection does check for CUDA, but there's no warning when
falling back to CPU on a large problem.

**Suggested fix**: Add a warning when solving on CPU with `N * V > threshold`
(e.g., 1M), or document expected solve times for different problem sizes in
the docstring. The GPU-hybrid path (precompute A^T A on CUDA, solve on CPU)
gives 44-194x speedups per the code comments.

## 5. No README or usage guide

**Encountered**: The only documentation is docstrings and type annotations.
For a public package, users have no entry point to understand the intended
workflow (collect → cap_vocab → find_weights → evaluate) or the PoE model.

**Workaround**: Read `__init__.py` docstring and test files to understand usage
patterns.

**Suggested fix**: Add a README.md with a quickstart example showing the
typical workflow: collect dictionary logprobs → cap vocabulary → fit weights →
evaluate → inspect support.

## 6. `LogprobCache` construction requires boilerplate

**Encountered**: Every notebook repeats the same pattern: load `.pt` file, define
prompt/input lists inline, manually construct `LogprobCache` with
`shared_indices=torch.arange(V)`. This is ~30 lines of setup per notebook.

**Workaround**: Copy-paste the construction pattern.

**Suggested fix**: Add `LogprobCache.from_file(path)` that loads a `.pt` file
containing the logprobs alongside prompts and test_inputs. The `save_cache` /
`load_cache` functions exist but the cached `.pt` files from the analysis scripts
use a different format than what `load_cache` expects.

## 7. All dataclasses use attribute access only -- no dict-style access

**Encountered**: All five dataclasses (`LogprobCache`, `Mixture`,
`DictionaryDiagnostics`, `SupportMetrics`, `GreedyStep`) require attribute
access (e.g., `sr.precision`). LLM-generated code and notebook authors
naturally try dict-style access (`sr['precision']`), especially for dataclasses
that feel like data containers.

**Workaround**: Use attribute access consistently.

**Suggested fix**: Either add `__getitem__` to the dataclasses that are
commonly used as return values (`SupportMetrics`, `DictionaryDiagnostics`,
`GreedyStep`), or make them NamedTuples which support both access styles.

## 8. `train_soft_prompt` computes loss on all tokens — no answer masking

**Encountered**: In notebook 12 (sequence memorization), training examples have the
structure `"The slice of the Hospitable Apprentice Sequence from index X to Y is [DIGITS]"`.
The preamble is ~100 tokens and the answer digits are ~20 tokens. Because
`train_soft_prompt` computes cross-entropy on the entire continuation
(lines 221-227 of `soft_prompt.py`: `shift_logits = logits[num_tokens:-1]`,
`shift_labels = input_ids[1:]`), the loss is dominated by the preamble. The soft
prompt learned to predict the query format fluently but achieved only 2.1% digit
accuracy — *below* the 10% random baseline.

**Workaround**: None within the current API. Would need to either:
- Manually modify the training loop to mask out non-answer tokens
- Restructure training data so the answer is the majority of tokens
- Use `train_soft_prompt_to_distribution()` instead (KL-based, not LM loss)

**Suggested fix**: Add a `loss_mask` or `answer_start_token` parameter to
`train_soft_prompt` so that loss is computed only on the answer portion. For
structured QA tasks, this is essential. Example API:
```python
train_soft_prompt(model, tokenizer, train_texts, answer_marker="is [",
                  loss_on="answer_only")
```

## 9. `generate()` has no greedy decoding mode

**Encountered**: When evaluating sequence memorization (notebook 12), deterministic
output was needed to measure digit accuracy. The `generate()` function always uses
`torch.multinomial` sampling (line 533 of `soft_prompt.py`), and `temperature=0.0`
would cause division-by-zero. There is no `do_sample=False` or `temperature=0`
(argmax) path.

**Workaround**: Use a moderately low temperature (e.g., 0.3) and hope for
consistency, or write a custom generation loop with `torch.argmax`.

**Suggested fix**: Add greedy decoding when `temperature <= 0` or add a
`do_sample: bool = True` parameter:
```python
if temperature <= 0 or not do_sample:
    next_token = logits.float().argmax().item()
else:
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    next_token = torch.multinomial(probs, 1).item()
```

## 10. `train_soft_prompt` silently drops short training examples

**Encountered**: Line 193-194 of `soft_prompt.py` filters out any training example
with fewer than 20 tokens (`if len(ids) >= 20`). This is silent — no warning is
emitted. In notebook 13 (sanity check), 100 French sentences averaging 21 tokens
were passed, but **28 were silently dropped** (72 survived). Similarly, 98 ALL CAPS
sentences lost 11 examples. The only clue is the printed example count buried in
training output (`Training soft prompt: ... 72 examples`), which is easy to miss.

**Workaround**: Ensure all training examples are at least 20 tokens. Manually check
token lengths before training.

**Suggested fix**: Either lower the threshold (the minimum meaningful length is
arguably much shorter), make it configurable via a `min_tokens` parameter, or emit
a warning when examples are dropped:
```python
if len(ids) < min_tokens:
    n_dropped += 1
    continue
if n_dropped and verbose:
    warnings.warn(f"Dropped {n_dropped}/{len(train_texts)} examples shorter "
                  f"than {min_tokens} tokens")
```

## 11. No held-out evaluation function for soft prompts

**Encountered**: In notebook 13 (sanity check), after training soft prompts on
French text, ALL CAPS text, and positive reviews, the only way to evaluate was
qualitative inspection of `generate()` output. There is no function to compute
loss/perplexity on held-out text with a trained soft prompt. The training loop
reports training loss per epoch, but there is no equivalent
`evaluate_soft_prompt(model, tokenizer, soft_prompt, test_texts) -> float`.

**Workaround**: Either inspect generation output qualitatively, or use the
decomposition pipeline (collect logprobs + measure KL) which is heavyweight for
a simple "did training work?" check.

**Suggested fix**: Add an evaluation function:
```python
def evaluate_soft_prompt(
    model, tokenizer, soft_prompt, texts, max_seq_len=64
) -> float:
    """Compute mean cross-entropy loss on texts with soft prompt prepended."""
```
This would mirror the training loss computation but on held-out data, giving
a quantitative measure of soft prompt effectiveness without needing the full
decomposition pipeline.
