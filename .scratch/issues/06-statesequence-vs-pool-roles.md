# Clarify the roles of `StateSequence` vs `SequencePool`

**Status:** `ready-for-agent`
**Type:** docs
**Source:** migrated from GitHub #10 (closed 2026-06-22)
**Related:** feeds `CONTEXT.md` (domain glossary) and ADR-0001

## Description

Both wrap a long-format `pl.DataFrame` with `(id, time, state)` columns:

- `StateSequence` — typed sequence container with format conversions
  (STS, SPS, DSS).
- `SequencePool` — adds a pre-extracted `dict[id, list[str]]` for fast random
  access, and exposes `compute_distances()`, `sample()`, `filter_by_length()`,
  `recode_states()`.

## Tasks

- [ ] Write the role distinction into `CONTEXT.md` (glossary) and/or docstrings.
- [ ] Ensure all `statistics/*` functions accept `StateSequence | SequencePool`
      uniformly (shared interface or duck typing on `.data`, `.alphabet`,
      `.config`).

> **Note:** the original issue mentioned `visualization/*` — that module was
> deleted in v0.4.0 (see ADR-0001), so it no longer applies.

## Comments
