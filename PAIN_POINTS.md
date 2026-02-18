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
