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
- [x] Ensure all `statistics/*` functions accept `StateSequence | SequencePool`
      uniformly (shared interface or duck typing on `.data`, `.alphabet`,
      `.config`). Done in 0.5.0 via the `core.protocols.SequenceData` protocol +
      `SequencePool.coerce` / `StateSequence.coerce`.

> **Note:** the original issue mentioned `visualization/*` — that module was
> deleted in v0.4.0 (see ADR-0001), so it no longer applies.

## Comments

- 2026-06-23 (architecture review, candidate B — Stage 1): introduced a
  symmetric coercion seam. `statistics.*` (31 functions across `descriptive`,
  `normative`, `transition`, `subsequence_mining`) now type their argument as
  `SequenceData` and normalize via `SequencePool.coerce` / `StateSequence.coerce`,
  replacing 16 inlined `isinstance` blocks in `descriptive`, a private `_get_pool`
  in `normative`, and several dead identical-arm `isinstance` branches. The
  duplication and the latent `get_sequence` return-type collision (excluded from
  the protocol) are gone. **Deferred to Stage 2:** the canonical-type decision
  (the evidence — `CLAUDE.md` data flow, metrics + clustering both on pool — leans
  toward `SequencePool` canonical, contradicting the old "StateSequence canonical"
  note; record an ADR either way) and the `CONTEXT.md`/docstring role write-up.
