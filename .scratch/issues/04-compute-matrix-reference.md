# Revisit `compute_matrix()` to confirm the reference is correct

**Status:** `needs-info` (blocked on the failing edge case)
**Type:** question
**Source:** migrated from GitHub #22 (closed 2026-06-22)
**Related:** [#01 OM substitution-matrix bug](01-om-subcost-matrix-bug.md) — likely the same defect (the "delta" dispatch hypothesis)

## Description

There was an edge case where `compute_matrix()` failed. Revisit to confirm
whether it is a real bug.

## Notes

Strong suspicion this is the same root cause as #01: `OptimalMatchingMetric`
not subclassing `SequenceMetric` changes the `compute_distances` /
`compute_matrix` dispatch path. Investigate together; if confirmed, fold this
into #01 and mark `wontfix`/duplicate here.

## Tasks

- [ ] Recover the failing edge case (input + traceback).
- [ ] Determine whether it is the OM dispatch bug (#01) or independent.

## Comments

- 2026-06-23 (architecture review, candidate A): `SequenceMetric.compute_matrix`
  was deleted (it had zero callers). The only matrix path now is
  `pool.compute_distances`. If the original `compute_matrix()` edge case was about
  the ABC path, it no longer exists; if it was about the pool path, it overlaps
  with #01. Still needs the failing case to confirm.
