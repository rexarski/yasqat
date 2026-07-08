# CONTEXT.md — yasqat domain glossary

Domain vocabulary for yasqat. Consumers: architecture reviews, bug diagnosis,
and feature work name modules and seams using *these* terms. (Seeded from
issue #06; issue #09 will grow this into the full glossary.)

## Containers

**Alphabet** — the ordered set of valid categorical states for a study
(e.g. `employed / unemployed / education / retired`), with optional colors
for downstream plotting. Shared by every container built from the same data.

**SequencePool** — the *canonical analysis container* (ADR-0002). Wraps the
long-format `(id, time, state)` DataFrame plus a pre-extracted
`dict[id, list[str]]` for fast random access. Every loader returns one;
`compute_distances()` (the metric dispatch seam), clustering, and statistics
consume it. Batch operations: `sample`, `filter_by_length`, `recode_states`,
`describe`.

**StateSequence** — the *representation view*. Same underlying DataFrame,
but oriented toward formats and per-sequence inspection: STS/SPS/DSS
conversions, `from_intervals` (interval-shaped input sampled onto a discrete
grid), per-sequence descriptives. Reached from a pool via
`pool.to_state_sequence()`.

**SequenceData** — the protocol both containers satisfy (`data`, `config`,
`alphabet`, `sequence_ids`). Public functions that accept sequences type their
argument as `SequenceData`; if they need methods beyond that surface they
normalize with `SequencePool.coerce` / `StateSequence.coerce` — the one
coercion seam (see CLAUDE.md house rule).

**DistanceMatrix** — the output of `compute_distances`: a symmetric pairwise
dissimilarity matrix over the pool's sequence ids, consumed by clustering
and discrepancy analysis. Consumers accept it or a raw numpy array through
the `DistanceMatrix.coerce` seam (mirroring `SequenceData` for containers).

## Sequence formats

**STS** — one row per (sequence, time point); the "wide spell-free" state
string form. **SPS** — state-permanence form: runs collapsed to
`state/duration` tokens. **DSS** — distinct-successive-states form: runs
collapsed to just the state, durations dropped.

## Analysis vocabulary

**Metric** — a free function `name_distance(seq_a, seq_b, **kwargs) -> float`
over integer-encoded arrays, registered in the dispatch dict in
`SequencePool.compute_distances`.

**Substitution matrix** — state×state cost table used by OM-family metrics;
built by `build_substitution_matrix` (constant, transition-rate, features…).
