# ADR-0001 — Unify the sequence model on `StateSequence`; drop visualization

**Status:** Accepted — shipped in v0.4.0 (2026-05-19)
**Supersedes:** the `BaseSequence` / `IntervalSequence` / `StateSequence` hierarchy
**Source:** brainstorm 2026-05-11 (was `docs/superpowers/specs/2026-05-11-v0.4.0-design.md`)

## Context

Through v0.3.x, yasqat carried three sequence types: a `BaseSequence` ABC, a
concrete `StateSequence`, and an `IntervalSequence` for interval-shaped input.
Loaders returned a `StateSequence | IntervalSequence` union and used
`infer_sequence_type()` to pick. This forced union types through `io/` and
`statistics/`, doubled the test surface, and exposed concepts (overlapping
intervals) that only one branch could ever produce. yasqat also shipped a
`visualization/` module (plotnine/matplotlib), which coupled a data library to a
specific plotting stack and pulled in heavy plotting dependencies.

Theme of the release: **less restraint on the consumer, more robust core** —
every change is a deletion or a consolidation.

## Decision

1. **Single sequence model.** `StateSequence` becomes the only sequence class —
   a concrete type with no inheritance hierarchy. Delete `BaseSequence` and
   `IntervalSequence`; drop `start_column`/`end_column` from `SequenceConfig`.
2. **Interval input via a classmethod.** Add
   `StateSequence.from_intervals(df, *, time_points=...)`, lifting the old
   interval→state-grid sampling (latest-start tiebreak, `join_asof(backward)`).
   For datetime start/end with no `time_points`, raise a clear `ValueError`
   instead of silently casting — no auto-generated grid.
3. **Extend the method surface** with `state_counts()`, `state_per_sequence()`,
   `duration()`, `total_duration_by_state()`, `spells_per_sequence()`, `span()`
   — all returning polars DataFrames.
4. **Delete the `visualization/` module** and its plotnine/matplotlib
   dependencies, tests, and `demo_showcase.ipynb`. Users plot the returned
   DataFrames with any tool.
5. **Delete `PatternCriterion`.** Use `QueryCriterion` or a direct polars
   expression instead.
6. **No deprecation cycle.** Pre-1.0 clean cut — removed names get a CHANGELOG
   note, no `DeprecationWarning` shim. Loaders narrow to `-> StateSequence`;
   `infer_sequence_type()` is removed.

## Consequences

- **Simpler core and IO:** no union types, no `infer_sequence_type`, one code
  path through `statistics/` and `clustering/`.
- **Breaking:** code importing `IntervalSequence`/`BaseSequence`, calling
  `to_state_sequence()`, or using `PatternCriterion` must migrate
  (`from_intervals`, `QueryCriterion`/polars). This is why the release was 0.4.0.
- **Lighter dependency tree:** plotnine/matplotlib gone.
- **Deferred:** the OM substitution-matrix bug was explicitly out of scope —
  tracked in `.scratch/issues/01-om-subcost-matrix-bug.md`.
