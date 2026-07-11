# Normative indicators — more methods

**Status:** `resolved` (0.5.0) — scope A + B shipped; C deferred
**Type:** enhancement
**Source:** migrated from GitHub #44 (closed 2026-06-22)
**Source file:** `src/yasqat/statistics/normative.py`
**Milestone:** 0.5.0

## Description

Add more method variants for the normative indicators, scoped against TraMineR.

- [x] Consult the TraMineR manual for additional variants of
      `proportion_positive`, `volatility`, `insecurity`, etc.
- [x] Scope against TraMineR's `seqistatd`, `seqibad`, `seqiprecarity`, etc.
- [x] Decide which to implement and pin expected values from TraMineR output.

## Triage (2026-07-11)

Lined the existing 7 indicators up against TraMineR's insecurity/integration
suite. Two gaps chosen for 0.5.0:

- **A — `individual_state_distribution()`** (TraMineR `seqistatd`): per-sequence
  count + proportion of time in each alphabet state, long-format DataFrame.
  Foundational; `proportion_positive`/`badness` are special cases of it.
- **B — `objective_volatility()`** (TraMineR `seqivolatility`): the *real*
  volatility measure — `w·pvisited + (1−w)·ptrans`, needing no pos/neg labels.
  Added as a NEW function; the existing sign-change `volatility()` is left
  untouched (renaming would break the 0.5.0 API for no user gain).

**Deferred — C (precarity/insecurity fidelity):** reworking `seqprecarity` /
`seqinsecurity` to TraMineR's ordered-badness-vector + complexity-correction
formulas is a larger, signature-changing job; parked for a later cycle, not
0.5.0.

## Comments

- 2026-07-11: **Resolved (0.5.0).** Shipped **A** `individual_state_distribution()`
  and **B** `objective_volatility()` in `statistics/normative.py`, exported from
  `statistics/__init__.py`, 12 new tests with values pinned by hand against the
  TraMineR definitions. `objective_volatility` is a NEW function; the existing
  sign-change `volatility()` is untouched (non-breaking). **C** (faithful
  `seqprecarity`/`seqinsecurity` with an ordered-badness vector) was deliberately
  deferred — larger, signature-changing; revisit in a later cycle. Full suite
  599 passed; ruff/format/mypy clean.
