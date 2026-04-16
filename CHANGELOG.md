# Changelog

## 0.3.2 (2026-04-16)

### Breaking changes

- **`Trajectory` and `TrajectoryPool` removed.** These types were unused in
  practice and added maintenance burden. Sequence-level analysis is fully
  covered by `StateSequence` / `IntervalSequence` + `SequencePool`.
- **`Alphabet.labels` removed.** Labels added complexity with no clear use
  case distinct from state names. `get_label()`, `with_labels()` are gone;
  `Alphabet` now only tracks states and colours.
- **`load_wide_format()` and `to_wide_format()` removed** from `yasqat.io`.
  Wide-format support was incomplete and rarely used.
- **`state_duration_stats()` renamed to `state_spell_stats()`** to accurately
  reflect that it counts consecutive spell runs, not clock-time durations.
- **`infer_sequence_type()` simplified** — now returns only `"state"` or
  `"interval"` (dropped `"event"` which was never reliably inferred).
- **`EventSequence` removed.** State and event data share an
  identical long-format column structure and were semantically
  indistinguishable in every downstream metric/statistic. Users should load
  event-style data directly as a `StateSequence`. `IntervalSequence` stays
  separate because it carries real extra structure (start/end columns).
- **`StateSequence.to_event_sequence()` and `StateSequence.to_interval_sequence()`
  removed.** `IntervalSequence.to_event_sequence()` is also gone.
  Use `StateSequence.to_sps()` for run-length/spell encoding.
- **`SequenceConfig.granularity` is now a string polars-truncate unit**
  Previously an integer bucket size; now a unit like `"1d"`,
  `"1w"`, `"1mo"` that truncates the `time_column` (and `start_column` /
  `end_column` on `IntervalSequence`) at construction time when those
  columns are datetime-typed. Raises `ValueError` if set against a
  non-datetime time column.
- **`modal_states(granularity=…)` is now string-only.**
  Integer bucket sizes are rejected with `TypeError`. Pass a polars
  `dt.truncate` unit like `"1d"` instead. Requires a datetime time column.
- **`SequencePool.to_wide_format()` / `to_long_format()` removed**
  They were never called in `src/`, not exported, and
  untested — matching the "no wide-format support in v0.3.2" promise.

### Bug fixes

- `Alphabet.index_of()` reverse lookup now works correctly.
- `Alphabet.get_color()` no longer returns colours for states not in the
  alphabet — raises `KeyError` instead.
- `Alphabet` constructors (`.from_sequence()`, `.from_series()`, direct init)
  now sort and deduplicate states automatically.
- `DistanceMatrix` now exposes a `.shape` property, fixing `AttributeError` in
  `pam_range()` and `extract_representatives()`.
- `hamming_distance()` raises `ValueError` for unequal-length sequences instead
  of silently producing wrong results.
- `optimal_matching_distance()` now validates that the substitution cost matrix
  dimensions match the pool alphabet, with a clear error message.
- `twed_distance()` error message corrected for unsupported substitution methods.
- `PatternCriterion` uses null-byte delimiters internally, preventing false
  matches when state names contain pattern characters (`-`, `*`, `+`, `?`).
- `PatternCriterion "A-?-B"` no longer returns empty when matched against a
  sequence `A→B` with zero middle states. The `?` / `*` / `+`
  wildcards now fold the delimiter inside the optional group.
- `load_dataframe()` validates that a user-supplied `Alphabet` covers all states
  actually present in the data.
- `BaseSequence.__init__` validates that any user-supplied alphabet covers
  every observed state, raising `ValueError` on mismatch.
- `optimal_matching_distance()` now rejects oversized substitution matrices
  too — the shape must match the alphabet exactly, not merely be ≥ it
.
- `PAMClustering.predict()` unwraps `DistanceMatrix` inputs, matching
  `pam_clustering()`. Previously raised `TypeError`.
- `dissimilarity_tree()` handles categorical (string/object) covariates via a
  one-vs-rest equality split, alongside the existing numeric threshold split
  Previously raised `UFuncTypeError`.
- `discrepancy_analysis()` now warns when perfect separation produces
  degenerate pseudo-R² = 1 / pseudo-F = 0 results.
- `discrepancy_analysis()` permutation p-value handles `pseudo_F = inf`
  correctly — only infinite permutation F values count as extreme
  Previously reported a spurious `p = 0.001` under perfect
  separation.
- `dissimilarity_tree()` uses improved defaults and enforces `min_node_size`,
  producing deeper and more useful trees.
- `index_plot(sort_by=…)` validates at runtime against
  `{"from.start", "from.end", "length", None}` and data columns.
  Previously silently fell through, masking config mistakes.
- `timeline_plot()` no longer renders blank: `y_pos` is cast to `Int64`
  Empty-data guard raises a clear `ValueError`.
- `modal_state_plot()` passes `granularity` through to `modal_states()` and
  guards against empty output with a descriptive error.
- `frequent_subsequences()` now returns a `pl.DataFrame` with columns
  `[subsequence, support, proportion]` instead of a list of dataclass objects.
- `OptimalMatchingMetric` is now exported from `yasqat.metrics`.
- `import yasqat` now makes `yasqat.io` available without a separate import.

### New features

- `sunburst_plot()` — concentric-ring visualisation showing hierarchical state
  transitions over time.
- `tree_plot()` — prefix-tree visualisation showing how sequences branch over
  time. Supports `min_support` pruning to focus on common paths.
- `PAMClustering.predict()` — assign new observations to existing medoids.
- `lcs_length()`, `lcs_similarity()`, `lcp_length()`, `lcp_similarity()`,
  `rlcp_length()`, `rlcp_similarity()` — convenience wrappers for
  length/similarity variants of LCS, LCP, and RLCP metrics.
- `mean_time_in_state()` now supports `per_sequence=True`.
- `state_distribution()` now supports `per_sequence=True`.
- `modal_states()` gains a `granularity` parameter for time-bucketed aggregation
  (e.g. day, week, month when working with timestamp-level data).
- `frequent_subsequences()` gains `min_length` parameter.
- `integration()` returns per-state values when `positive_states` is not
  specified (previously returned a single aggregate).
- `subsequence_count()` gains `states_filter` (restrict to specific states) and
  `use_log` (log-transform counts to prevent overflow on large pools).
- `SequencePool.compute_distances()` gains `n_jobs` parameter for parallel
  pairwise computation.
- `twed_distance()` accepts a `lam` kwarg alias for `lmbda`.
- `k_range(start, end)` helper for inclusive ranges; `pam_range()` auto-expands
  2-tuples with a warning and the parameter is renamed `k_values` (legacy
  `k_range=` kwarg still accepted).
- `RepresentativeResult.__repr__` now previews `indices` and `scores`; the
  docstring includes a recipe for resolving indices back to sequences
.

### Performance

- `longitudinal_entropy()` rewritten with polars `group_by` — single-pass
  vectorized computation replaces per-sequence Python loop.
- `turbulence()` rewritten with polars `group_by` over spell data.
- `normalized_turbulence()` uses polars join + expressions instead of row-by-row
  iteration.
- `entropy_plot()` vectorized with Polars expressions.
- `IntervalSequence.has_overlaps()` vectorized with Polars (was O(n²) pairwise).
- `StartsWithCriterion` vectorized with polars `group_by` (was slow Python loop).
- `PatternCriterion` vectorized via polars `group_by + str.concat + str.contains`
  — the regex compile is hoisted out of the per-sequence loop.
- `IntervalSequence.to_state_sequence(time_points=…)` rewritten with
  `polars.join_asof`. Drops complexity from
  O(n_sequences · n_intervals · n_time_points) to roughly
  O((n_intervals + n_time_points) · log). The tiebreaker when multiple
  intervals cover the same time point is now "latest start wins"
  (previously incidental row order).

### Removals

- Quarto documentation site removed from repository. Documentation will be
  rebuilt with Sphinx in a future release.
- `.to_pandas()` conversions across `src/yasqat/visualization/` removed
  — plotnine ≥ 0.13 consumes polars frames directly via the
  dataframe-interchange protocol.

## 0.3.1 (2026-03-09)

### Performance

- `SequencePool._extract_sequences()` replaced with O(N) `group_by().agg()`
  approach (was O(N×U) — one full table scan per unique ID). Significant
  speedup for large datasets (50k+ sequences).
- `SequencePool.describe()` uses null-safe accessors for length statistics.

### New features

- `load_dataframe()` in `yasqat.io` — build a `SequencePool` directly from a
  polars DataFrame. Recommended entry point for Hive/Spark/Arrow workflows.
  Supports `drop_nulls` parameter for cleaning null states on ingestion.
- `transition_rates()` and `transition_rate_matrix()` accept `exclude_self`
  parameter to filter out trivial self-transitions (A→A). Defaults to `False`.
- All visualization functions (`index_plot`, `distribution_plot`, `timeline_plot`,
  `spell_duration_plot`) auto-suppress legends when the alphabet has >15 states.
  Pass `show_legend=True` to override.

### Removals

- TanaT source code removed (was already unused in v0.3.0).

## 0.3.0 (2026-02-23) — first public release

### Breaking changes

- `SequencePool.compute_distances()` now returns `DistanceMatrix` (with `.values`
  and `.labels`) instead of a raw `np.ndarray`.
- `sequence_ids` is now a `@property` on all `BaseSequence` subclasses
  (`StateSequence`, `EventSequence`, `IntervalSequence`). Call it without
  parentheses: `seq.sequence_ids` not `seq.sequence_ids()`.
- `IntervalSequence.to_state_sequence()` now returns `StateSequence` instead of
  `pl.DataFrame`.

### Bug fixes

- Clustering functions (`pam_clustering`, `clara_clustering`,
  `hierarchical_clustering`) now correctly unwrap `DistanceMatrix` input via
  `.values` (previously checked for a non-existent `.matrix` attribute, silently
  passing the wrapper object through instead of the underlying array).
- `yasqat.__version__` now reads the installed version from `importlib.metadata`
  instead of a hardcoded string (`"0.1.0"`) that disagreed with `pyproject.toml`
  (`"0.2.1"`).

### Additions

- `SequenceConfig` is now exported from `yasqat` and `yasqat.core`.
- Integration test suite added (`tests/test_integration.py`), covering the full
  pool → distances → clustering → quality workflow.
