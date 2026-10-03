# OM substitution-matrix bug

**Status:** `resolved` (dev, 2026-09-29 — not reproducible; regression tests added)
**Type:** bug
**Source:** migrated from `docs/superpowers/specs/2026-05-11-om-subcost-investigation.md` (2026-06-22); deferred from the v0.4.0 brainstorm
**Related:** [#04 compute_matrix() reference](04-compute-matrix-reference.md) — likely the same defect
**Source files:** `src/yasqat/metrics/optimal_matching.py`, `src/yasqat/metrics/base.py`, `src/yasqat/core/pool.py`

## Symptom

> "`om_distance` with subcost matrix still has an error."

No traceback or input was supplied. Fix is deferred until a sharp repro exists.

## Pre-investigation findings

1. **`OptimalMatchingMetric` does not inherit from `SequenceMetric`.** It is a
   bare class exposing only `compute(seq_a, seq_b)`, with no `pairwise()` /
   `compute_matrix()` override — yet `CLAUDE.md` requires every metric to
   subclass `SequenceMetric`. Likely consequence:
   `pool.compute_distances(OptimalMatchingMetric(...))` takes a different code
   path (or fails) than other metrics.
2. The square/large-enough validation block was touched in v0.3.2 (hot-fix B2);
   the *messaging* was refined, but the math may still be wrong in some case.

## Hypotheses (walk in order when a repro arrives)

- **alpha — propagation.** Pool-level `compute_distances` doesn't propagate the
  `sm` matrix; per-pair calls see the default `sub_cost`.
- **beta — encoding drift.** Per-pair encoding produces state indices that
  disagree with the matrix the user built.
- **gamma — message only.** Math correct; validation message misleading.
- **delta — dispatch.** `OptimalMatchingMetric` not being a real
  `SequenceMetric` subclass causes the wrong path inside `compute_distances`.

Most likely: **delta** (structural) feeding **alpha** (propagation).

## Tasks

- [ ] Obtain or construct a minimal failing repro.
- [ ] Add a failing test in `tests/test_metrics/test_optimal_matching.py`.
- [ ] Likely fix: make `OptimalMatchingMetric(SequenceMetric)`, plus whatever
      the repro reveals.
- [ ] CHANGELOG entry under the version current at fix time.

## Comments

- 2026-06-23 (architecture review, candidate A): the **delta hypothesis is now
  moot** — `OptimalMatchingMetric` and the whole metric class layer were deleted,
  so there is no longer a divergent class path. If the subcost error persists it
  must live in the surviving free-function/pool path (alpha/beta/gamma) — still
  blocked on a repro. Status stays `needs-info`.
- 2026-09-29: Systematic repro hunt on dev: `sm` passed to
  `compute_distances("om", sm=...)` as an ndarray, list of lists, int
  matrix, float32 matrix, asymmetric matrix, a matrix from
  `substitution_cost_matrix("trate")`, from `build_substitution_matrix`,
  with `sub_cost` also set, with `n_jobs=2`, on a pool with a wider
  declared alphabet, on string ids, and through a coerced `StateSequence`
  — all agree with a per-pair brute force; a too-small matrix raises the
  documented `ValueError`. The only failure found was `sm="trate"` as a
  *string*, which the wrapper rejected; that now resolves via the engine's
  `prepare` hook. Regression tests: `tests/test_metrics/test_engine.py::
  TestSubstitutionMatrixByName`. Reopen with a concrete input if the
  original error reappears.
