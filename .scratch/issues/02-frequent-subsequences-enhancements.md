# `frequent_subsequences()` enhancements

**Status:** `ready-for-agent`
**Type:** enhancement (the bug portion is already fixed)
**Source:** migrated from GitHub #43 (closed 2026-06-22)
**Source file:** `src/yasqat/statistics/subsequence_mining.py`

## Background

Originally filed because the documented example crashed:

```python
patterns = frequent_subsequences(pool, min_support=0.05, max_length=3)
patterns.filter(...)   # used to error
```

**Already resolved (v0.3.2+):** return type is `pl.DataFrame`, so `.filter()`
works. The `min_length` parameter also already exists. So the bug and the
`min_length` ask are done.

## Remaining work

- [ ] Add association-rule measures to the output: confidence, lift, leverage,
      conviction (cross-reference TraMineR / standard sequence-rule definitions).
- [ ] Decide the output shape (extra columns vs a separate rules DataFrame).
- [ ] Tests pinning each measure on a small hand-computable example.
- [ ] Docs/example update.

## Comments

- 2026-06-08: bug + `min_length` verified already shipped; only the
  association-rule measures remain.
