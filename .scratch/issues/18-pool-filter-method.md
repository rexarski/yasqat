# `filter_sequences` returns a DataFrame, so "filter then score" needs a manual pool rebuild

**Status:** `resolved` (dev, 2026-09-17)
**Type:** enhancement
**Source:** surfaced while verifying the VP-deck code sample, 2026-09-08
**Source file:** `src/yasqat/filters/criteria.py`, `src/yasqat/core/pool.py`

## Description

Every `statistics.*` function accepts a pool through the `coerce` seam, but
`filter_sequences` hands back a plain polars DataFrame. The natural chain
"define a situation, then mine it" therefore needed three lines:

```python
kept = filter_sequences(pool, criteria)
situation = SequencePool(kept, config=pool.config, alphabet=pool.alphabet)
rules = association_rules(situation, ...)
```

Passing the DataFrame straight into `association_rules` raises
`AttributeError: 'DataFrame' object has no attribute 'config'`. The slide-8
code sample in the VP deck shipped that way until it was caught.

## Resolution

Additive, no breaking change: `SequencePool.filter(criteria, combine="and")`
applies the same criteria via `filter_sequences` and returns a pool with the
parent's config and **full** alphabet (so encodings stay comparable and a
state filtered out of every sequence is still a known state). Chains
directly into `compute_distances` and any `statistics` function.

`filter_sequences` keeps its DataFrame return for callers who want rows; its
docstring now points at `pool.filter` as the typed path. Tests in
`tests/test_core/test_pool.py::TestPoolFilter` pin ids, alphabet
preservation, the empty case, and a distance-matrix chain.
