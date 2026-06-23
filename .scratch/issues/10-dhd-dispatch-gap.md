# DHD has no pool-level matrix path

**Status:** `needs-triage`
**Type:** enhancement
**Source:** surfaced by the architecture review (candidate A), 2026-06-23

## Description

`dhd_distance` is not registered in `SequencePool.compute_distances`'s dispatch
dict, so there is no pool-level way to build a DHD distance matrix. SoftDTW had
the same gap and was registered in 0.5.0; DHD was deliberately left out because
it has structural preconditions that don't fit the generic
`compute_distances(method=str, **kwargs)` shape.

## Why DHD is awkward for the generic dispatch

- DHD requires **equal-length sequences** (`dhd_distance` raises on unequal).
- It needs a `position_costs` array of shape `(T, n_states, n_states)` where `T`
  is the shared sequence length, built via `build_position_costs`. A pool-wide
  call needs `T` to match every sequence's length.

## Options

- [ ] Register `"dhd"` in the dispatch and document the equal-length +
      `position_costs` precondition (error clearly when unmet).
- [ ] Add a dedicated `pool.compute_dhd(position_costs=...)` helper instead.
- [ ] Leave DHD free-function-only and document it as such.

## Comments
