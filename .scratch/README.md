# yasqat issue tracker

Local-markdown tracker — the **single source of truth** for yasqat issues and
in-progress work. Conventions: `docs/agents/issue-tracker.md`. Status vocabulary:
`docs/agents/triage-labels.md`. Architecture decisions: `docs/adr/`.

Migrated from GitHub Issues on **2026-06-22** — the 8 open issues were closed
there with a pointer to this tracker.

## Board

| # | Issue | Type | Status |
|---|-------|------|--------|
| 01 | [OM substitution-matrix bug](issues/01-om-subcost-matrix-bug.md) | bug | `needs-info` |
| 02 | [`frequent_subsequences()` enhancements](issues/02-frequent-subsequences-enhancements.md) | enhancement | `ready-for-agent` |
| 03 | [Normative indicators — more methods](issues/03-normative-indicators.md) | enhancement | `needs-triage` |
| 04 | [Revisit `compute_matrix()` reference](issues/04-compute-matrix-reference.md) | question | `needs-info` |
| 05 | [`encode_states` / `recode_states` — keep or remove?](issues/05-encode-recode-states.md) | question | `resolved` (0.5.0) |
| 06 | [StateSequence vs SequencePool roles](issues/06-statesequence-vs-pool-roles.md) | docs | `resolved` (0.5.0) |
| 07 | [`sequence_frequency_table` vs `subsequence_count`](issues/07-freqtable-vs-subseqcount-docs.md) | docs | `resolved` (0.5.0) |
| 08 | [Compile `pitch.md`](issues/08-pitch-doc.md) | docs | `ready-for-human` |
| 09 | [Compile v0.3.2 terminology glossary](issues/09-v032-appendix-glossary.md) | docs | `resolved` (0.5.0) |
| 10 | [DHD has no pool-level matrix path](issues/10-dhd-dispatch-gap.md) | enhancement | `resolved` (0.5.0) |
| 11 | [More time-series content](issues/11-time-series-content.md) | enhancement | `needs-triage` |
| 12 | [More subsequence mining](issues/12-more-subsequence-mining.md) | enhancement | `needs-triage` |
| 13 | [New documentation website](issues/13-documentation-website.md) | docs | `needs-triage` |
| 14 | [Mind map of functions](issues/14-function-mind-map.md) | docs | `needs-triage` |

## Status legend

- `needs-triage` — needs evaluation / scoping
- `needs-info` — blocked, waiting on a repro or more detail
- `ready-for-agent` — fully specified; an agent can pick it up with no extra context
- `ready-for-human` — needs human implementation or judgment
- `wontfix` — will not be actioned

## Conventions in brief

- One file per backlog issue: `issues/NN-slug.md`, numbered from `01`.
- Multi-issue features get their own dir: `<feature-slug>/PRD.md` +
  `<feature-slug>/issues/NN-slug.md`.
- Triage state is the `**Status:**` line near the top of each file.
- Keep this board's table in sync when you add an issue or change a status.
