# yasqat — Code Review Progress

Started: 2026-02-23
Branch: `dev`
Reviewer: external Python package critic

---

## Review summary

An external review identified 8 issues. One claim was wrong and one additional bug
was found during verification. All actionable items have been resolved.

---

## Issues

| # | Area | Severity | Status |
|---|------|----------|--------|
| 1 | Version mismatch (`__init__.py` vs `pyproject.toml`) | High | ✅ Fixed |
| 2 | `SequencePool.compute_distances()` returns `np.ndarray` not `DistanceMatrix` | High | ✅ Fixed |
| B | Clustering `.matrix` attribute bug (should be `.values`) — found in verification | Critical | ✅ Fixed |
| 3 | `IntervalSequence.to_state_sequence()` returns `pl.DataFrame` not `StateSequence` | Medium | ✅ Fixed |
| 4 | `sequence_ids` inconsistency: property on Pool, method on BaseSequence | Medium | ✅ Fixed |
| 5 | O(n²) Python loop in `compute_distances` (no batch path) | Medium | ✅ Documented |
| 6 | `SequenceConfig` not in public API | Low | ✅ Fixed |
| 7 | Filter `.filter()` returns `pl.DataFrame` — no wrapping guidance | Low | ✅ Documented |
| 8 | Missing integration / property-based tests | Low | ✅ Fixed (integration) |
| ~~ | pyarrow claimed optional — **incorrect**: viz uses `to_pandas()` in 9 places | — | No action needed |

---

## Changes made

### 1. Version — single source of truth
**File:** `src/yasqat/__init__.py`

Replaced hardcoded `__version__ = "0.1.0"` with a runtime lookup via
`importlib.metadata.version("yasqat")`, with a `PackageNotFoundError` fallback to
`"unknown"`. The installed version from `pyproject.toml` (`0.2.1`) is now the single
source of truth. Verified: `import yasqat; yasqat.__version__` returns `"0.2.1"`.

### B. Clustering DistanceMatrix extraction bug (`.matrix` → `.values`)
**Files:** `src/yasqat/clustering/pam.py`, `clara.py`, `hierarchical.py`

All three clustering entry-points checked `hasattr(distance_matrix, "matrix")` to
unwrap a `DistanceMatrix`, but the dataclass attribute is `values`. Fixed to check
`hasattr(distance_matrix, "values")` and extract `.values` accordingly. Also
standardised to `np.asarray(..., dtype=np.float64)` in `clara` and `hierarchical`.

### 2. `SequencePool.compute_distances()` returns `DistanceMatrix`
**Files:** `src/yasqat/core/pool.py`, `tests/test_core/test_pool.py`

Changed return type from `np.ndarray` to `DistanceMatrix(values=distances, labels=ids)`.
Updated three distance tests to assert `isinstance(dm, DistanceMatrix)` and access
`.values` for numpy assertions.

### 3. `IntervalSequence.to_state_sequence()` returns `StateSequence`
**Files:** `src/yasqat/core/sequence.py`, `tests/test_core/test_sequence.py`

Changed both return sites (empty case and normal case) to wrap in
`StateSequence(data=df, config=self._config, alphabet=self._alphabet)`. Updated the
test that accessed `.columns` and `.sort()` directly on the result to go through
`.data`.

### 4. `sequence_ids` unified as `@property`
**Files:** `src/yasqat/core/sequence.py`, `core/trajectory.py`,
`filters/criteria.py`, `tests/test_core/test_sequence.py`

`BaseSequence.sequence_ids` was an abstract method; `SequencePool.sequence_ids` was
already a `@property`. Converted the abstract declaration and all three concrete
implementations (`StateSequence`, `EventSequence`, `IntervalSequence`) to
`@property`. Updated all callers to drop the `()`:
- `criteria.py`: 3 call sites
- `trajectory.py`: 6 call sites
- `sequence.py` itself: 2 call sites (inside `IntervalSequence`)
- `tests/test_core/test_sequence.py`: 1 call site

### 5. O(n²) loop — documented
**File:** `src/yasqat/core/pool.py`

Added a `Note:` section in `compute_distances()` docstring pointing users toward
`.sample()` or CLARA for large pools.

### 6. `SequenceConfig` exported in public API
**Files:** `src/yasqat/core/__init__.py`, `src/yasqat/__init__.py`

Added `SequenceConfig` to imports and `__all__` in both files.

### 7. Filter return type — documented
**File:** `src/yasqat/filters/criteria.py`

Added a wrapping example to the docstrings of `SequenceCriterion.filter()` and
`filter_sequences()` showing how to construct a `StateSequence` from the returned
`pl.DataFrame`.

### 8. Integration tests added
**File:** `tests/test_integration.py`

Six end-to-end tests in `TestFullWorkflow`:
- `test_pool_to_distance_matrix` — shape, symmetry, zero diagonal, `DistanceMatrix` type
- `test_label_lookup_on_distance_matrix` — `get_distance(id, id) == 0` for all IDs
- `test_clustering_accepts_distance_matrix` — PAM accepts `DistanceMatrix` directly
- `test_quality_after_clustering` — silhouette scores in `[-1, 1]`, positive mean for separable data
- `test_statistics_on_pool` — `longitudinal_entropy` and `complexity_index` return correct shapes
- `test_condensed_form_roundtrip` — `to_condensed` / `from_condensed` preserves values and labels

---

## Test counts

| Session | Tests |
|---------|-------|
| Before fixes | 526 passed |
| After all fixes | 532 passed (+6 integration) |

---

## Nothing remaining

All review items have been resolved. The `pyarrow` claim was verified incorrect
(visualization modules call `to_pandas()` in 9 places across 5 files; the dependency
is legitimately required).

Property-based / hypothesis-style tests were out of scope for this session. If desired,
they can be added in a follow-up under `tests/test_properties/`.
