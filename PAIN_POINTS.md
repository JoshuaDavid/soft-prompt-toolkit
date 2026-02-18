# Pain Points

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

**Status**: Wontfix — design decision to keep dataclasses as pure containers.

## 8. `train_soft_prompt` computes loss on all tokens — no answer masking

**Encountered**: In notebook 12 (sequence memorization), training examples have the
structure `"The slice of the Hospitable Apprentice Sequence from index X to Y is [DIGITS]"`.
The preamble is ~100 tokens and the answer digits are ~20 tokens. Because
`train_soft_prompt` computes cross-entropy on the entire continuation,
the loss is dominated by the preamble. The soft prompt learned to predict the
query format fluently but achieved only 2.1% digit accuracy — *below* the 10%
random baseline.

**Workaround**: Restructure training data so the answer is the majority of tokens,
or use `train_soft_prompt_to_distribution()` instead (KL-based, not LM loss).

**Status**: Wontfix — specialized QA use case; users needing answer-only loss
should write a custom training loop.

## 12a. `train_soft_prompt` sequential batching (4-16x throughput waste)

**Encountered**: Profiled the training loop on an RTX 4090 with Qwen3-4B (8 GB
in bf16). The bottleneck is **memory bandwidth** — at batch_size=1 with ~36 tokens,
the arithmetic intensity is only 36 FLOPs/byte, far below the RTX 4090's
compute/bandwidth crossover of ~358 FLOPs/byte. Each forward pass spends most of
its time reading 8 GB of model weights from HBM, not computing.

The inner training loop processes each example as a separate batch_size=1 forward
pass. With batch_size=4, this reads the 8 GB model weights **4 separate times**
instead of once:

```
Forward bs=1:  47 ms     Forward bs=4:  47 ms (same!)
4x sequential: 188 ms    Batched:       47 ms
```

Profiled end-to-end (72 examples, 15 epochs, RTX 4090):

```
Approach                        Batch  Epoch    Total   Speedup
Current (sequential bs=1)          4   10.1s   151.2s     1.0x
Batched fwd+bwd                    4    2.5s    36.8s     4.1x
Batched fwd+bwd                    8    1.3s    18.9s     8.0x
Batched fwd+bwd                   16    0.7s    10.2s    14.9x
```

**Fix**: Pad sequences to equal length, use attention masks, run a single batched
forward pass per optimizer step.

## 13. Soft prompt training requires prefix-diverse training data

**Encountered**: In notebook 14, a soft prompt trained on ALL-CAPS text
produced 0% CAPS when `generate()` used a lowercase prefix like `" The"`.
Inspecting logits revealed the soft prompt correctly shifted predictions toward
CAPS tokens (96.6% CAPS mass with no prefix, 99.7% after `" THE"`), but a
single lowercase prefix token completely overrode this (0.2% CAPS after `" The"`).

The root cause is autoregressive conditioning: the model's next-token prediction
is dominated by the most recent tokens. A soft prompt 20 positions back provides
a prior, but the prefix token provides direct autoregressive evidence that
overpowers it. Since training data was all CAPS, the SP learned
P(caps|soft_prompt, caps_context) but never learned
P(caps|soft_prompt, lowercase_context).

**Fix**: Training data must include the prefix diversity the SP will encounter at
inference time. Training on mixed-case data (lowercase prefix → CAPS
continuation) with 60 diverse starters (both cased variants + common words)
achieved 97-99% CAPS mass across all test prefixes, including previously
impossible ones like `" The"` (0.2% → 99.4%) and `" Yesterday"` (7.5% → 98.6%).

**Implication**: This applies to all style properties (language, tone,
formatting), not just case. Homogeneous training data teaches *continuation*
of a style, not *forcing* of a style.

**Status**: Not a toolkit bug — training data design responsibility. But the
documentation should warn users that training data prefix diversity matters.
