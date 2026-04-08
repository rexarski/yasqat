# Changelog

## 0.3.2 (unreleased)

### Bug fixes

- `frequent_subsequences()` now returns a `pl.DataFrame` with columns
  `[subsequence, support, proportion]` instead of a list of dataclass objects.
  The `subsequence` column is a `list[str]`, enabling polars filtering like
  `results.filter(pl.col("subsequence").list.len() == 2)`.
- `OptimalMatchingMetric` is now exported from `yasqat.metrics` (previously
  only the function `optimal_matching_distance` was re-exported).

### Performance

- `longitudinal_entropy()` rewritten with polars `group_by` pipeline — replaces
  per-sequence Python loop and `collections.Counter` with a single-pass
  vectorized computation.
- `turbulence()` rewritten with polars `group_by` over spell data — eliminates
  per-sequence DataFrame filtering.
- `normalized_turbulence()` uses polars join + expressions instead of row-by-row
  iteration.

### New features

- `sunburst_plot()` — concentric-ring visualisation showing hierarchical state
  transitions over time. Each ring represents a time position; arc widths are
  proportional to the number of sequences following each path.
- `tree_plot()` — prefix-tree visualisation showing how sequences branch over
  time. Supports `min_support` pruning to focus on the most common paths.

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
