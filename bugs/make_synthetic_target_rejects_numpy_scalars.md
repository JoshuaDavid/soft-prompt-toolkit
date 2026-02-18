# Bug: `make_synthetic_target` rejects numpy scalar types in dict weights

## Description

`make_synthetic_target(cache, weights)` accepts `dict[int, float]` as the
`weights` parameter, but beartype validation rejects numpy scalar types
(`np.int64`, `np.float64`) which are what numpy operations naturally produce.

## Reproduction

```python
import numpy as np
from soft_prompt_toolkit import make_synthetic_target

rng = np.random.RandomState(42)
K = 100

# This is the natural way to build a sparse weight dict from numpy
indices = rng.choice(K, size=5, replace=False)  # np.int64 array
values = rng.randn(5) * 0.3                      # np.float64 array
weights = {indices[i]: values[i] for i in range(5)}

# type(list(weights.keys())[0])   -> np.int64
# type(list(weights.values())[0]) -> np.float64

target = make_synthetic_target(cache, weights)
# BeartypeCallHintParamViolation: ... dict[int, float] ...
```

## Expected behavior

The function should accept numpy integer and floating-point scalar types
in the dict, since they are the natural output of numpy operations and are
semantically equivalent to Python builtins.

## Actual behavior

beartype raises `BeartypeCallHintParamViolation` because `np.int64 is not int`
and `np.float64 is not float`.

## Suggested fix

Either:
1. Widen the type annotation: `dict[int | np.integer, float | np.floating]`
2. Add an internal cast at the top of the function:
   ```python
   if isinstance(weights, dict):
       weights = {int(k): float(v) for k, v in weights.items()}
   ```

Option 2 is preferred as it's backwards-compatible and handles all numpy
scalar subtypes without expanding the public type signature.
