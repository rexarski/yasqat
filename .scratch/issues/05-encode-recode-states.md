# `encode_states` / `recode_states` — keep or remove?

**Status:** `resolved` (in 0.5.0-unreleased)
**Type:** question (API surface decision)
**Source:** migrated from GitHub #15 (closed 2026-06-22)
**Source files:** `src/yasqat/core/sequence.py`, `src/yasqat/core/pool.py`

## Description

- `StateSequence.encode_states()` returned a flat 1D `np.ndarray` of integer
  indices for all rows (no per-sequence boundaries).
- `SequencePool.recode_states(mapping)` renames/merges states via
  `pl.col.replace()`. Useful for analysts collapsing categories.

## Resolution (architecture review, candidate C)

- **`encode_states` removed.** It had zero callers and its flat all-rows shape
  was never used by the metrics (which encode per-sequence via
  `SequencePool.get_encoded_sequence()`). `get_encoded_sequence` is now the
  single encoding seam. Users wanting indices from a `StateSequence` call
  `seq.alphabet.encode(seq.data[state_col].to_list())`.
- **`recode_states` kept** — analyst-facing, the supported rename/merge path.

CHANGELOG entry under `## 0.5.0 (unreleased)` → Breaking changes.

## Comments
