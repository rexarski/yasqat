# OM substitution-matrix bug — deferred investigation

**Status:** open · deferred from v0.4.0 brainstorm on 2026-05-11
**Owner:** rexarski
**Source files:** `src/yasqat/metrics/optimal_matching.py`, `src/yasqat/metrics/base.py`, `src/yasqat/core/pool.py`

## Symptom (per user)

> "`om_distance` with subcost matrix still has an error."

No specific traceback or input was supplied during the v0.4.0 brainstorm; the
fix is deferred until a sharp repro is in hand.

## Pre-investigation reading

`optimal_matching.py` end-to-end. Two suspicious facts surfaced:

1. **`OptimalMatchingMetric` does not inherit from `SequenceMetric`.** It is a
   bare class exposing only `compute(seq_a, seq_b)`. There is no
   `compute_matrix()` / `pairwise()` override. `CLAUDE.md` declares "Every
   metric must subclass `SequenceMetric`" — most metrics do, OM does not.
   Likely consequence: `pool.compute_distances(OptimalMatchingMetric(...))`
   takes a different code path (or fails) than the other metrics.
2. The square / large-enough validation block was already touched in v0.3.2
   (hot-fix B2). The error *messaging* was refined; the math may still be
   wrong in some specific case.

## Hypotheses to investigate

When a repro arrives, walk these in order:

- **alpha — propagation.** Pool-level `compute_distances` does not propagate
  the `sm` matrix correctly. Per-pair calls see the default `sub_cost` instead
  of the user matrix. **Check:** instrument `_optimal_matching_kernel` and
  confirm `sub_costs` is the expected matrix on each call from a `pool`-level
  invocation.
- **beta — encoding drift.** Per-pair encoding inside the pool produces state
  indices that disagree with the matrix the user built. State-index ordering
  drift between alphabet builds. **Check:** compare `pool.alphabet.encode(states)`
  index ordering against the matrix the user passed.
- **gamma — message only.** Underlying math is correct; the validation message
  is misleading. **Check:** compute by hand, confirm distance matches kernel.
- **delta — dispatch.** `OptimalMatchingMetric` not being a real
  `SequenceMetric` subclass causes the wrong code path inside
  `pool.compute_distances`. **Check:** trace `compute_distances` for both a
  proper `SequenceMetric` subclass (e.g. `HammingMetric`) and
  `OptimalMatchingMetric` and diff the path.

Most likely culprit on prior reading: **delta** (structural) feeding into
**alpha** (propagation). Beta is plausible if the user is mixing matrices
built from different pools. Gamma is the lowest-effort outcome.

## When to pick this up

After v0.4.0 ships, or sooner if the user supplies a repro. Either way the
investigation should land its own PR with:
- A failing test in `tests/test_metrics/test_optimal_matching.py` that pins
  the bug.
- The minimal fix (likely making `OptimalMatchingMetric` inherit from
  `SequenceMetric`, plus whatever the repro reveals).
- A CHANGELOG entry under whatever version is current at fix time.

## Cross-reference

- `src/yasqat/metrics/optimal_matching.py:121-138` — current validation block.
- `src/yasqat/metrics/optimal_matching.py:150` — class declaration that
  should `class OptimalMatchingMetric(SequenceMetric):` but does not.
- `src/yasqat/metrics/base.py:87` — `SequenceMetric` ABC.
- v0.3.2 hot-fix B2 — prior validation refinement.
