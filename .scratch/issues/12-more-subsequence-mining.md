# More subsequence mining

**Status:** `needs-triage`
**Type:** enhancement
**Source:** user todo list, 2026-07-07
**Related:** issue 02 (association-rule measures for `frequent_subsequences` —
already `ready-for-agent`; that item is likely the first slice of this one)
**Source file:** `src/yasqat/statistics/subsequence_mining.py`

## Description

Grow the mining module beyond plain frequent-subsequence discovery.
Candidate directions, to be picked at triage:

- **Association-rule measures** — confidence, lift, leverage, conviction
  (this is issue 02; do it first or fold it in here).
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
