# `encode_states` / `recode_states` — keep or remove?

**Status:** `needs-triage`
**Type:** question (API surface decision)
**Source:** migrated from GitHub #15 (closed 2026-06-22)
**Source files:** `src/yasqat/core/sequence.py`, `src/yasqat/core/pool.py`

## Description

- `StateSequence.encode_states()` returns a flat 1D `np.ndarray` of integer
  indices for all rows. Used internally by `SequencePool.get_encoded_sequence()`
  for metrics.
- `SequencePool.recode_states(mapping)` renames/merges states via
  `pl.col.replace()`. Useful for analysts collapsing categories.

## Decision needed

- [ ] `encode_states`: keep public, or make internal? (It's an implementation
      detail of the metrics path.)
- [ ] `recode_states`: keep (it's analyst-facing and useful).
- [ ] Document the outcome and update `__all__` accordingly.

## Comments
