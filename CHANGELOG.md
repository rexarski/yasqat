# Changelog

## 0.5.0 (unreleased)

### Breaking changes

- **Metric class layer removed.** The `SequenceMetric` ABC and the per-metric
  classes (`HammingMetric`, `LCSMetric`, `LCPMetric`, `RLCPMetric`, `DTWMetric`,
  `SoftDTWMetric`, `TWEDMetric`, `Chi2Metric`, `EuclideanMetric`, `DHDMetric`,
  `OptimalMatchingMetric`) are deleted. No metric ever subclassed the ABC and its
  `compute_matrix()` had no callers — the live path is the free `*_distance`
  functions dispatched by `SequencePool.compute_distances(method=...)`, which
  stays. `DistanceMatrix` and `build_substitution_matrix` remain in
  `yasqat.metrics`. Migrate `SomeMetric(...).compute(a, b)` →
  `some_distance(a, b, ...)`, and matrix computation →
  `pool.compute_distances(method=...)`.
- **File loaders now return `SequencePool`.** `load_csv`, `load_json`, and
  `load_parquet` return a `SequencePool` (matching `load_dataframe`) instead of
  a `StateSequence`, and route through `load_dataframe` internally — one
  DataFrame→pool seam for validation and alphabet checks. `SequencePool` is now
  the canonical analysis container (ADR-0002); `StateSequence` is the
  representation view for format conversions. Migrate
  `seq = load_csv(...)` + `StateSequence`-only calls to
  `load_csv(...).to_state_sequence()`. The `save_*` functions accept either
  container (typed as `SequenceData`).
- **`StateSequence.encode_states()` removed.** It had no callers and returned a
  flat, sequence-boundary-less array of every row's state index — a shape the
  metrics never used (they encode per-sequence via
  `SequencePool.get_encoded_sequence()`). To integer-encode states from a
  `StateSequence`, call `seq.alphabet.encode(seq.data[state_col].to_list())`
  directly. `SequencePool.recode_states()` is unaffected and remains the
  supported way to rename/merge states.

### New features

- **`"softdtw"` is now selectable in `SequencePool.compute_distances(...)`.**
  Registering it in the dispatch closed a gap where SoftDTW had no pool-level
  matrix path.
- **`"dhd"` is now selectable in `SequencePool.compute_distances(...)`.** The
  position-dependent cost array is built from the pool via
  `build_position_costs` when not passed explicitly (`position_costs=...`);
  unequal-length pools raise a clear `ValueError` (DHD requires equal
  lengths). Closes the last metric with no pool-level matrix path.
- **`SequencePool.coerce()` and `StateSequence.coerce()` classmethods.** Each
  normalizes any sequence container (`StateSequence` or `SequencePool`) to its
  own type — identity if already that type, otherwise rebuilt from the shared
  `data`/`config`/`alphabet`. These are the single seam through which
  `statistics.*` accepts either container.
- **`DistanceMatrix.coerce()` — one seam for distance-matrix inputs.** Every
  distance consumer (`pam_clustering`, `clara_clustering`,
  `hierarchical_clustering`, the `clustering.quality` functions,
  `discrepancy_analysis`, `multi_factor_discrepancy`, `dissimilarity_tree`)
  now accepts a `DistanceMatrix` or a raw numpy array, normalized through the
  new classmethod. Discrepancy analysis and dissimilarity trees previously
  accepted only raw arrays, forcing `dm.values` unwrapping after
  `compute_distances`; five inlined `hasattr(x, "values")` unwrap blocks are
  gone. Coerced inputs get 2-D/square validation for free.
- **`filters.*` accept either sequence container.** The criterion classes and
  `filter_sequences` type their argument as the `SequenceData` protocol, so a
  `SequencePool` (what loaders now return) works directly — no conversion to
  `StateSequence` needed. The protocol gained `sequence_ids` (both containers
  already exposed it with identical semantics); the criteria need no runtime
  coercion at all.
- **`association_rules()` — sequential association rules from mined patterns.**
  New function in `yasqat.statistics` that splits each frequent subsequence
  into an ordered antecedent (prefix) and consequent (suffix) and reports the
  standard measures — `confidence`, `lift`, `leverage`, `conviction` — as a
  separate rules `DataFrame`. Marginal supports are read straight from the
  frequent-set, which the Apriori property guarantees already contains every
  prefix and suffix, so no extra data scan is needed. Closes the last open
  item of the `frequent_subsequences` enhancement request.
- **`individual_state_distribution()` — per-sequence state distribution.** New
  function in `yasqat.statistics` (TraMineR `seqistatd`) reporting, for each
  sequence and each alphabet state, the number of time units spent in that
  state and its proportion of the sequence length (unvisited states included
  with a zero count). Returns a long-format `DataFrame`.
- **`objective_volatility()` — label-free volatility.** New function in
  `yasqat.statistics` (TraMineR `seqivolatility`) measuring
  `w · pvisited + (1 − w) · ptrans` — a weighted blend of alphabet coverage
  and transition rate that, unlike the existing `volatility()`, needs no
  positive/negative state labels.

### Documentation

- **Sphinx documentation site** (`docs/`), replacing the removed Quarto site.
  autodoc + napoleon (Google-style docstrings) + MyST (Markdown) on the furo
  theme; an API reference generated per subpackage from each `__all__`, and
  glossary/changelog pages that render `CONTEXT.md` / `CHANGELOG.md` in place
  (single source of truth). Builds and deploys to GitHub Pages on push to
  `main` via `.github/workflows/docs.yml`; install with the new `docs` extra.
- `CONTEXT.md` domain glossary at the repo root: container roles (ADR-0002),
  sequence formats, sequence anatomy, and analysis vocabulary.
- "When to use which" cross-references in `sequence_frequency_table`
  (complete-trajectory frequencies) vs `subsequence_count`
  (within-trajectory variety).

### Internal

- **`statistics.*` now accept the `SequenceData` protocol via one coercion
  seam.** The 31 functions across `descriptive`, `normative`, `transition`, and
  `subsequence_mining` previously each inlined an `isinstance` coercion (16 in
  `descriptive` alone) or routed through a private `_get_pool`. They now type
  their argument as `core.protocols.SequenceData` and normalize via
  `SequencePool.coerce` / `StateSequence.coerce`. Behaviour is unchanged; the
  duplicated coercion and several dead identical-arm `isinstance` branches are
  gone. (Progresses the `StateSequence`/`SequencePool` role cleanup; the
  canonical-type decision is deferred.)
- **`statistics.*` per-sequence functions share one reduce seam.** The 17
  functions in `descriptive`/`normative` that mapped a scalar over each
  sequence and returned either a per-sequence `DataFrame` or an aggregate each
  re-implemented the same loop + return-shape contract. They now delegate to a
  private `reduce_per_sequence(sequence, fn, name, per_sequence, aggregate)`
  helper; each function keeps only its per-sequence scalar. Behaviour and
  signatures are unchanged (net −226 lines). The `float | DataFrame` contract
  and the mean/sum collapse now live in one place.
- **The twin `coerce` classmethods share one rebuild rule.**
  `StateSequence.coerce` and `SequencePool.coerce` had byte-identical bodies;
  both now delegate to `core.protocols.coerce_container`, so the
  `SequenceData` field set is spelled out once.

## 0.4.1 (2026-06-22)

### Bug fixes

- **`generate_markov_sequences` is now importable from `yasqat.synthetic`.**
  It was documented and tested but missing from the package's public exports
  (`__all__`), so `from yasqat.synthetic import generate_markov_sequences`
  failed.

### Documentation

- Removed stale `plotnine` references from the package docstring and README
  left over after the v0.4.0 visualization removal.
- `CLAUDE.md` is now tracked in-repo and slimmed to development house rules
  (the per-module API catalogue was dropped as drift-prone).
- Added `docs/adr/` (architecture decision records); ADR-0001 records the
  v0.4.0 sequence-model unification and visualization removal.

### Internal

- Adopted an in-repo issue tracker under `.scratch/` as the single source of
  truth; migrated and closed the project's GitHub issues. Agent-skills
  configuration lives in `docs/agents/`.
- Bumped `astral-sh/setup-uv` from v6 to v8.2.0 in CI (Node 24). Pinned to a
  concrete release because the action publishes no floating `v8` major tag.

## 0.4.0 (2026-05-19)

### Breaking changes — sequence model unification

- **`IntervalSequence` removed.** Interval-shaped input is now sampled into
  a `StateSequence` via the new `StateSequence.from_intervals(df, time_points=...)`
  classmethod. Every time point in a `StateSequence` carries exactly one state.
- **`BaseSequence` ABC removed.** `StateSequence` is the only sequence class.
- **`SequenceConfig.start_column` and `SequenceConfig.end_column` removed.**
  These only applied to the deleted `IntervalSequence`.
- **`infer_sequence_type()` removed.** No longer meaningful with a single
  sequence type.
- **Loader signatures simplified.** `load_csv`, `load_json`, `load_parquet`
  no longer take a `sequence_type` parameter and always return `StateSequence`.
  `save_csv` / `save_json` / `save_parquet` parameter types narrowed
  similarly.
- **`StateSequence.intervals_per_sequence` is `spells_per_sequence`** — same
  return shape, name now matches the TraMineR "spell" terminology and the
  unified model.

### Breaking changes — visualization module removed

- **`yasqat.visualization` removed in full.** `index_plot`, `timeline_plot`,
  `distribution_plot`, `entropy_plot`, `modal_state_plot`, `mean_time_plot`,
  `parallel_coordinate_plot`, `sunburst_plot`, `tree_plot`, and
  `spell_duration_plot` are all gone.
- **`plotnine` and `matplotlib` removed from dependencies.** Users now feed
  the polars DataFrames returned by yasqat methods (`to_sps()`,
  `state_distribution()`, `state_counts()`, etc.) directly into their tool
  of choice (matplotlib, altair, observable, …). `Alphabet.colors` is
  retained as a hand-off for consistent palette use.
- **`demo_showcase.ipynb` removed.**

### Breaking changes — filters

- **`PatternCriterion` removed.** The dual-grammar (simple wildcards + regex)
  with its dual-separator hack was fragile. Users needing pattern matching
  use polars expressions directly. Migration example:

  ```python
  # Old:
  PatternCriterion(pattern="A-*-C")
  # New (polars expression):
  matched_ids = (
      seq.data
      .group_by("id")
      .agg(pl.col("state").str.join("-").alias("s"))
      .filter(pl.col("s").str.contains(r"^A-[^-]+-C$"))
      .get_column("id")
      .to_list()
  )
  ```

  This is more code than `PatternCriterion("A-*-C")` was. The tradeoff is
  "verbose but transparent" instead of "concise but fragile".

### New methods on `StateSequence`

- **`state_counts()`** — pool-wide row count per state.
- **`state_per_sequence(*, proportion: bool = False)`** — per-sequence state
  distribution. Counts by default; pass `proportion=True` for within-sequence
  shares.
- **`duration()`** — thin alias over `to_sps()` returning
  `[id, state, duration]`.
- **`total_duration_by_state()`** — sum spell durations per state.
- **`spells_per_sequence()`** — distinct run-length spells per sequence
  (replaces the old `IntervalSequence.intervals_per_sequence`).
- **`span()`** — `[id, first, last, span]` per sequence. Works for integer
  and datetime time columns; for datetime, `span` is a `pl.Duration`.

### New constructor

- **`StateSequence.from_intervals(data, *, time_points=..., ...)`** —
  build a `StateSequence` by sampling interval-shaped data on a grid.
  Lifts the `join_asof` machinery from the deleted
  `IntervalSequence.to_state_sequence`. Latest-start tiebreak preserved.
  Datetime input requires explicit `time_points` (raises `ValueError`
  otherwise — there is no obvious default granularity).

### CI

- Bumped `actions/checkout@v4` → `@v5` and `astral-sh/setup-uv@v5` → `@v6`
  to stay on Node 24 ahead of the June 2026 Node 20 deprecation.

### Deferred

- An OM (`optimal_matching_distance`) bug with substitution matrices is
  tracked separately at
  `.scratch/issues/01-om-subcost-matrix-bug.md`. Pre-investigation
  reading flags `OptimalMatchingMetric` not subclassing `SequenceMetric`
  as the most likely structural culprit. Fix awaits a sharp repro.

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
