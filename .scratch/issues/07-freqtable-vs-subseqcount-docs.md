# Clarify `sequence_frequency_table()` vs `subsequence_count()`

**Status:** `resolved` (0.5.0)
**Type:** docs
**Source:** migrated from GitHub #41 (closed 2026-06-22)

## Description

These serve different purposes and the difference should be documented clearly:

- `sequence_frequency_table()` — counts **full-sequence** frequencies.
- `subsequence_count()` — counts **sub-patterns**.

## Tasks

- [x] Add a clear "when to use which" note to the docstrings of both functions.
- [x] Optionally add a short example contrasting the two outputs.

## Comments

- 2026-07-07 (resolved, 0.5.0): both docstrings got a symmetric "When to use
  which" paragraph — `sequence_frequency_table` counts complete-trajectory
  frequencies across the pool (TraMineR `seqtab`); `subsequence_count`
  measures within-trajectory variety (the phi ingredient of turbulence) —
  each cross-referencing the other, with a shared `[A-B, A-B, A-C]`
  contrasting example.
