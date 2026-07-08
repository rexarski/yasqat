# CONTEXT.md — yasqat domain glossary

Domain vocabulary for yasqat. Consumers: architecture reviews, bug diagnosis,
and feature work name modules and seams using *these* terms. (Seeded from
issue #06; expanded into the full glossary per issue #09.)

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

## Sequence anatomy

**State** — one categorical value from the alphabet, observed at one time
point (e.g. `unemployed` at month 7).

**Sequence** — the ordered series of states for one unit of observation
(one person, one patient), identified by its id.

**Spell** — a maximal run of the same state (`A A A B` has an A-spell of
length 3 and a B-spell of length 1). Spells are what SPS/DSS compress;
`spell_count` and `spells_per_sequence` count them.

**Transition** — a change of state between adjacent time points;
`transition_rate_matrix` estimates state→state transition probabilities,
which also seed the transition-rate substitution costs.

## Analysis vocabulary

**Metric** — a free function `name_distance(seq_a, seq_b, **kwargs) -> float`
over integer-encoded arrays, registered in the dispatch dict in
`SequencePool.compute_distances`.

**Optimal Matching (OM)** — the workhorse edit-distance metric: the minimal
cost of turning one sequence into another using substitutions (priced by the
substitution matrix) and indels. Variants (OMloc, OMspell, OMstran, …) reweight
localization, spells, or transitions.

**Indel** — an insertion/deletion edit operation in OM-family metrics, priced
by the `indel` parameter; the counterweight to substitution cost.

**Substitution matrix** — state×state cost table used by OM-family metrics;
built by `build_substitution_matrix` (constant, transition-rate, features…).
DHD generalizes it to a per-position `(T, n_states, n_states)` array
(`build_position_costs`).

**Longitudinal entropy** — Shannon entropy of a sequence's state
distribution: 0 when one state fills the whole sequence, maximal when time
is spread evenly across states.

**Turbulence** — Elzinga's composite of a sequence's distinct-subsequence
count (phi, see `subsequence_count`) and the variance of its spell
durations; high for trajectories that switch often and irregularly.

**Complexity index** — Gabadinho's blend of entropy and the number of
transitions, an alternative instability measure to turbulence.

**Discrepancy analysis** — distance-based pseudo-ANOVA
(`discrepancy_analysis`): decomposes the sum of squared pairwise distances
into within/between-group parts, yielding pseudo-R² (share of distance
variance a grouping explains) and pseudo-F, with permutation p-values.
`dissimilarity_tree` applies the same criterion recursively to grow a
covariate-split tree.

**Normative indicators** — indicators that presuppose a quality ordering on
states (positive vs negative): volatility, precarity, insecurity,
degradation, badness, integration. They live in `statistics/normative.py`
and require the user to say which states count as positive.
