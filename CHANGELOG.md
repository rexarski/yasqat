# Changelog

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
