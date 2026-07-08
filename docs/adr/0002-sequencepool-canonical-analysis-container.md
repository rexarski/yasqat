# ADR-0002 — `SequencePool` is the canonical analysis container

**Status:** Accepted — targeted at v0.5.0 (unreleased), decided 2026-07-07
**Refines:** ADR-0001 (which unified the sequence *model* on `StateSequence`
but did not rule on the analysis container)
**Source:** architecture review 2026-06-22, candidate B (Stage 2);
`.scratch/issues/06-statesequence-vs-pool-roles.md`

## Context

yasqat has two containers over the same long-format `(id, time, state)`
DataFrame:

- `StateSequence` — format conversions (STS/SPS/DSS), interval sampling
  (`from_intervals`), per-sequence descriptives.
- `SequencePool` — pre-extracted per-id state lists, `compute_distances()`
  (the metric dispatch seam), `sample()`, `filter_by_length()`,
  `recode_states()`.

Stage 1 of candidate B (0.5.0, commit `ee6f5c6`) removed the
`StateSequence | SequencePool` unions from `statistics/*` via the
`core.protocols.SequenceData` protocol and symmetric `coerce` classmethods.
What remained undecided was which container the *data-flow spine* produces
and consumes.

The evidence pointed one way:

- `metrics` and `clustering` operate exclusively on `SequencePool`.
- `CLAUDE.md`'s data flow reads: io loaders → `SequencePool` → metrics →
  `DistanceMatrix` → clustering / statistics.
- The io seam was split: `load_dataframe` returned `SequencePool`, but
  `load_csv` / `load_json` / `load_parquet` returned `StateSequence` — whose
  users then could not call `compute_distances` without a manual hop. The
  v0.4.1 README quick-start bug (fixed in `0565835`) was a direct symptom.

An older design note ("make `StateSequence` canonical") predated the pool's
growth into the analysis engine; ADR-0001's "unify on StateSequence" was
about deleting the `BaseSequence`/`IntervalSequence` hierarchy, not about
this choice.

## Decision

1. **`SequencePool` is the canonical container.** It is what loaders return
   and what the analysis pipeline (metrics, clustering, statistics) consumes.
2. **`StateSequence` is the representation view** — format conversions,
   interval sampling, per-sequence descriptives. Reached from a pool via
   `pool.to_state_sequence()`, or accepted anywhere via
   `StateSequence.coerce`.
3. **All loaders return `SequencePool`.** `load_csv`, `load_json`, and
   `load_parquet` now route through `load_dataframe`, making it the single
   DataFrame→pool seam (validation, alphabet checks, null handling live
   there once).
4. **Functions that accept sequences keep typing the argument as
   `SequenceData`** and normalizing with `coerce` (the CLAUDE.md house
   rule) — the canonical choice does not narrow public argument types.
   `save_csv` / `save_json` / `save_parquet` accordingly accept either
   container.

## Consequences

- **Breaking (0.5.0):** code doing `seq = load_csv(...)` and then calling
  `StateSequence`-only methods must add one hop:
  `pool.to_state_sequence()`. Pre-1.0 clean cut, CHANGELOG note, no shim
  (same policy as ADR-0001).
- The README quick-start loses its `SequencePool.coerce` workaround — the
  root cause is retired.
- `filters/` still types its argument as `StateSequence`. Widening it to
  `SequenceData` is a natural follow-up but is deliberately out of scope
  here; tracked as future work, not decided by this ADR.
- Future loaders (e.g. Arrow, database readers) must return `SequencePool`
  and should route through `load_dataframe`.
