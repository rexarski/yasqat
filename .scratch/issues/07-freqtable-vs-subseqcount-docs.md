# Clarify `sequence_frequency_table()` vs `subsequence_count()`

**Status:** `ready-for-agent`
**Type:** docs
**Source:** migrated from GitHub #41 (closed 2026-06-22)

## Description

These serve different purposes and the difference should be documented clearly:

- `sequence_frequency_table()` — counts **full-sequence** frequencies.
- `subsequence_count()` — counts **sub-patterns**.

## Tasks

- [ ] Add a clear "when to use which" note to the docstrings of both functions.
- [ ] Optionally add a short example contrasting the two outputs.

## Comments
