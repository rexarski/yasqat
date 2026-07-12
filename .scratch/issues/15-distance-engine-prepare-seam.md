# Split the pairwise distance engine out of SequencePool

**Status:** `needs-triage`
**Type:** enhancement / architecture
**Source:** architecture review 2026-07-11, candidate B (deferred from the
0.5.0 review — structural move judged too close to the release)
**Related:** the metric dispatch seam (`CLAUDE.md` "Metrics" rule), issue 01/04
(OM subcost repro would exercise the same seam)

## Description

`SequencePool.compute_distances` (`core/pool.py:163–275`) holds three
responsibilities: the dispatch dict (clean), per-metric preparation, and the
O(n²) pairwise engine (sequential + threaded). The preparation step leaks:

```python
if method == "dhd" and "position_costs" not in kwargs:
    kwargs["position_costs"] = build_position_costs(self)
```

The pool knows a metric's internals; adding another metric that needs
pool-derived setup means editing `pool.py`, not just `metrics/`. OM's
substitution matrix is prepared *outside* this seam (caller-supplied kwarg),
so the prep story is asymmetric.

**Proposed shape:** move the engine to `metrics/` (e.g. a
`compute_matrix(pool, method, n_jobs, **kwargs)` free function that
`compute_distances` delegates to), and give each metric an optional uniform
`prepare(pool, kwargs)` hook (default no-op) so a metric supplies its own
pool-derived setup behind one seam. `compute_distances` keeps its public
signature and simply delegates — non-breaking for callers.

## Why deferred

Touches the metric dispatch seam named in `CLAUDE.md` ("register it in the
dispatch dict in `SequencePool.compute_distances`"). The house rule would need
rewording, and moving the engine mid-release invites regressions in the one
hot path every metric shares. Better as the opening move of the next cycle,
where the OM subcost repro (issues 01/04) can be re-run against the new seam.

## Tasks

- [ ] Triage: confirm the `prepare(pool, kwargs)` hook shape vs. alternatives
      (e.g. a per-metric dataclass carrying both `fn` and `prepare`).
- [ ] Move the O(n²) driver (sequential + threaded) into `metrics/`.
- [ ] Give `dhd` (and OM's substitution matrix) a symmetric prepare path.
- [ ] Reword the `CLAUDE.md` "Metrics" dispatch rule to match.

## Comments

- 2026-07-11: Filed from the architecture review. Candidate A (statistics
  reduce seam) and C (unified `coerce`) shipped in the same review; B and D
  (this + issue 16) deferred as structural/API moves too close to 0.5.0.
