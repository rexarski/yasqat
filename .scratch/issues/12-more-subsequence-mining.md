# More subsequence mining

**Status:** `needs-triage`
**Type:** enhancement
**Source:** user todo list, 2026-07-07
**Related:** issue 02 (association-rule measures for `frequent_subsequences` —
**resolved in 0.5.0**; shipped as `association_rules()`. That was the first
slice of this issue; the remaining directions below are still open.)
**Source file:** `src/yasqat/statistics/subsequence_mining.py`

## Description

Grow the mining module beyond plain frequent-subsequence discovery.
Candidate directions, to be picked at triage:

- ~~**Association-rule measures** — confidence, lift, leverage, conviction
  (this is issue 02).~~ **Done in 0.5.0** — `association_rules()`.
- **Closed / maximal frequent subsequences** — prune the redundant output
  that plain frequent-pattern enumeration produces.
- **Gap and window constraints** — max-gap / max-window parameters on
  matching (TraMineR `seqefsub` constraints).
- **Discriminant subsequences** — which subsequences best separate groups
  (TraMineR `seqecmpgroup`); pairs naturally with discrepancy analysis.
- **Event-sequence view** — mining on transition events rather than state
  spells.

## Tasks

- [ ] Triage: pick and order the concrete slices with the user.
- [ ] Decide the relationship to issue 02 (absorb vs sequence after it).
- [ ] Anchor each slice to its TraMineR reference before implementation.

## Comments
