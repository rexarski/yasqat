# `frequent_subsequences()` enhancements

**Status:** `resolved` (0.5.0)
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

- [x] Add association-rule measures to the output: confidence, lift, leverage,
      conviction (cross-reference TraMineR / standard sequence-rule definitions).
- [x] Decide the output shape (extra columns vs a separate rules DataFrame).
- [x] Tests pinning each measure on a small hand-computable example.
- [x] Docs/example update.

## Comments

- 2026-06-08: bug + `min_length` verified already shipped; only the
  association-rule measures remain.
- 2026-07-10: **Resolved (0.5.0).** New `association_rules()` function in
  `yasqat.statistics.subsequence_mining` (separate rules `DataFrame`, not extra
  columns on `frequent_subsequences` — a rule's antecedent⇒consequent shape and
  per-split cardinality don't fit one-row-per-subsequence). Reports confidence,
  lift, leverage, conviction; `min_confidence` filter; conviction is `inf` for
  exact rules. Marginal supports read from the mined frequent-set (Apriori
  guarantees every prefix/suffix is present — no extra scan). Shared
  `_mine_frequent` helper now backs both functions. Eight new tests pin B⇒C
  (confidence 2/3, lift 8/9, leverage −0.0625, conviction 0.75) and the
  infinite-conviction edge case. All gates green (587 tests).
