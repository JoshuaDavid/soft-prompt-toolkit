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
