# Bug: `train_soft_prompt` loss accumulation causes OOM on consumer GPUs

## Description

`train_soft_prompt` accumulates loss across batch elements before calling
`backward()`, which retains all computation graphs simultaneously. On a 24 GB
GPU (RTX 4090) with Qwen3-4B (8 GB in bf16), this causes OOM at batch_size >= 2.

## Reproduction

```python
from soft_prompt_toolkit import train_soft_prompt

# With Qwen3-4B on a 24 GB GPU:
sp, losses = train_soft_prompt(
    model, tokenizer, train_texts,
    num_tokens=20, epochs=15, batch_size=4,  # OOMs during backward
)
```

The OOM occurs at `soft_prompt.py` line 231 (`batch_loss.backward()`), not during
the forward passes. The forward passes at lines 212-227 each create a computation
graph that is retained until backward is called. With batch_size=4, four
computation graphs for a 4B model exhaust GPU memory.

## Measured impact

On RTX 4090 (24 GB) with Qwen3-4B (8 GB bf16):

| Retained graphs | Activations | backward() result |
|-----------------|-------------|-------------------|
| 1               | 0.23 GB     | OK (1.01 GB peak) |
| 2               | ~0.46 GB    | **OOM**           |
| 4               | ~0.92 GB    | **OOM**           |

With gradient accumulation (1 graph at a time):

| Batch size | Peak memory | Result |
|------------|-------------|--------|
| 8          | 1.01 GB     | OK     |
| 32         | 1.01 GB     | OK     |

## Root cause

Lines 210-231 of `soft_prompt.py`:

```python
batch_loss = torch.tensor(0.0, device=device)
for idx in batch_indices:
    # ... forward pass creating computation graph ...
    batch_loss = batch_loss + loss  # retains graph!
batch_loss = batch_loss / len(batch_indices)
batch_loss.backward()  # must backprop through ALL graphs simultaneously → OOM
```

The issue is that `batch_loss + loss` keeps all previous computation graphs alive,
because autograd needs them for the eventual backward pass. The backward pass
itself also needs additional temporary memory proportional to model_size × n_graphs.

## Fix

Replace loss accumulation with gradient accumulation (mathematically equivalent):

```python
for idx in batch_indices:
    # ... forward pass ...
    loss = F.cross_entropy(...) / len(batch_indices)
    loss.backward()  # immediate backward, computation graph freed
# gradients are accumulated in soft_prompt.parameters().grad
optimizer.step()
optimizer.zero_grad()
```

This is a 3-line change. Peak memory becomes constant regardless of batch size.

## Why this hasn't been caught

The training may appear to work if:
- The GPU has enough memory (e.g., 48 GB A6000)
- Sequences are very short (fewer activations)
- batch_size=1 is used (only 1 graph, no accumulation)
- PyTorch memory allocator happens to have enough reserved memory from fragmentation

In our testing, notebook 13 ran with batch_size=4 on a freshly loaded model (no
fragmentation), which sometimes succeeds. But the profiling session (with prior
allocations) consistently OOMed at batch_size >= 2.
