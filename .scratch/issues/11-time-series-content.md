# More time-series content

**Status:** `on-hold`
**Type:** enhancement
**Source:** user todo list, 2026-07-07
**Milestone:** deferred (parked 2026-07-11 — no longer targeting 0.5.0)

## Description

Expand yasqat's time-oriented analysis surface. Deliberately unscoped —
"time series content" needs a triage conversation to pin down which of these
(or something else) is meant:

- **Cross-sectional statistics over time** — state distributions, entropy, or
  modal states evaluated *per time point* across the pool (TraMineR's
  `seqstatd` transversal statistics). `state_distribution` /
  `modal_states` partially cover this; audit the gaps.
- **Time-varying analyses** — rolling-window turbulence/complexity, spell
  survival curves, time-to-first-transition.
- **Datetime ergonomics** — richer handling of timestamp time columns
  (resampling, alignment, calendar granularity beyond `modal_states`'
  `granularity` param).
- **Alignment tooling** — aligning sequences on an event (e.g. "months since
  first employment") rather than absolute time.

## Tasks

- [ ] Triage: pick the concrete scope with the user; split into sub-issues if
      more than one area is wanted.
- [ ] Cross-reference the TraMineR function(s) that anchor each chosen item.

## Comments

- 2026-07-11: **On hold** (user's call). Scope stays deliberately unscoped —
  parked until a future cycle rather than triaged now. Does not block 0.5.0.
