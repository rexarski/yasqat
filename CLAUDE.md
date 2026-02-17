# CLAUDE.md — yasqat development guide

## Project identity

**yasqat** (Yet Another Sequence Analytics Toolkit) is a modern Python library for categorical sequence analysis built on **polars** and **plotnine**. It targets social-science and life-course research workflows — labour-market trajectories, health pathways, educational histories, etc.

- Python ≥ 3.11, packaged with **hatchling**
- Source layout: `src/yasqat/`
- Tests: `tests/` (pytest, mirrors `src/` structure)
- Demo notebook: `demo_showcase.ipynb`

## Reference implementations

yasqat draws on two established packages for feature parity and correctness:

| Package | Language | Role |
|---------|----------|------|
| **TraMineR** | R (CRAN) | Gold standard for sequence analysis. Primary reference for algorithm correctness, naming conventions, and feature scope. |
| **TanaT** | Python | Earlier Python port of TraMineR. Useful for understanding Python-specific adaptations, but yasqat aims to surpass it with polars performance and modern API design. |

When implementing new features, cross-reference both packages for algorithmic correctness. TraMineR documentation is the canonical source for expected behaviour.

## Architecture

```
src/yasqat/
├── core/           # Alphabet, Sequence types, SequencePool, Trajectory
├── metrics/        # Pairwise distance metrics (OM, Hamming, LCS, DTW, …)
├── statistics/     # Transition rates, descriptive stats, discrepancy, subsequence mining
├── clustering/     # PAM, CLARA, hierarchical, quality indices
├── visualization/  # plotnine-based plots (index, timeline, distribution, modal, parallel)
├── filters/        # Sequence filtering criteria
├── io/             # Data loaders (CSV, parquet → polars)
└── synthetic/      # Synthetic data generators (financial/Markov)
```

### Key data flow

1. Raw data → `io.loaders` → `pl.DataFrame`
2. DataFrame → `core.Alphabet` + `core.SequencePool` (or `Trajectory`/`TrajectoryPool`)
3. Pool → `metrics.*` → `DistanceMatrix` (numpy array wrapper)
4. DistanceMatrix → `clustering.*` or `statistics.*`
5. Any of the above → `visualization.*` → plotnine `ggplot` objects

## Coding conventions

### Style

- **Formatter/linter**: ruff (line-length 88, target py311)
- **Type checking**: mypy strict mode (relaxed for `visualization.*` and `metrics.*` due to untyped deps)
- Every file starts with `from __future__ import annotations`
- Use `TYPE_CHECKING` guard for imports only needed by type hints
- Prefer `@dataclass` (or `@dataclass(frozen=True)`) for value types
- Use `__all__` in `__init__.py` to control public API

### Naming

- Modules: `snake_case.py`, one concept per file (e.g. `hamming.py`, `optimal_matching.py`)
- Classes: `PascalCase` (e.g. `SequencePool`, `HammingMetric`, `DistanceMatrix`)
- Functions: `snake_case` (e.g. `hamming_distance`, `generate_markov_sequences`)
- Metric classes inherit from `SequenceMetric` (ABC in `metrics/base.py`)
- Each metric module typically exports both a low-level function and a class

### Dependencies

- **polars** for DataFrames (never pandas in core code; pyarrow bridge only for interop)
- **numpy** for numeric arrays and distance matrices
- **numba** (`@njit`) for performance-critical inner loops in metrics
- **plotnine** for all visualization (ggplot2 grammar)
- **scipy** for hierarchical clustering linkage

### Error handling

- Raise `ValueError` for invalid inputs with clear messages
- Validate at public API boundaries, trust internal calls
- Use `assert` only in tests, never in library code

## Testing conventions

- Framework: **pytest** with `--strict-markers`
- Test structure mirrors source: `tests/test_metrics/test_hamming.py` ↔ `src/yasqat/metrics/hamming.py`
- Test classes: `class TestFeatureName:` with descriptive method names (`test_identical_sequences`, `test_single_mismatch`)
- All test methods have return type annotation `-> None`
- Shared fixtures go in `tests/conftest.py`
- Use `np.testing.assert_allclose` for floating-point comparisons
- Use `pytest.raises` for expected exceptions

## Common commands

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_metrics/test_hamming.py

# Run with coverage
uv run pytest --cov=yasqat --cov-report=term-missing

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/yasqat/
```

## Module-level guidance

### metrics/

- Every metric must subclass `SequenceMetric` from `metrics/base.py`
- Implement `compute(seq_a, seq_b) -> float` and `pairwise(pool) -> DistanceMatrix`
- Use `@numba.njit` for the inner distance computation when possible
- `DistanceMatrix` wraps a symmetric numpy array with optional labels

### statistics/

- Functions accept `SequencePool` or `pl.DataFrame` and return polars DataFrames
- `transition.py`: transition rate matrices
- `descriptive.py`: entropy, turbulence, complexity
- `discrepancy.py`: pseudo-R² measures for cluster quality
- `disstree.py`: dissimilarity tree (CART-like splitting on distance matrices)

### visualization/

- Every plot function returns a `plotnine.ggplot` object (composable, not side-effecting)
- Follow plotnine idioms: `aes()`, `geom_*`, `theme_*`
- Use `Alphabet.colors` for consistent state-to-colour mapping across plots

### clustering/

- PAM and CLARA operate on `DistanceMatrix`
- Hierarchical wraps `scipy.cluster.hierarchy`
- `quality.py`: silhouette, ASW, Hubert's C — all accept `DistanceMatrix` + cluster labels

### core/

- `Alphabet`: immutable (`frozen=True`), defines valid states + colours + labels
- `StateSequence` / `EventSequence` / `IntervalSequence`: different sequence representations
- `SequencePool`: collection of sequences with shared alphabet (primary analysis unit)
- `Trajectory` / `TrajectoryPool`: time-stamped sequence variants

## Adding a new feature

1. Identify the corresponding TraMineR/TanaT function for reference
2. Create module in appropriate subpackage (`snake_case.py`)
3. Implement with type hints and docstrings
4. Export from subpackage `__init__.py`
5. Write tests mirroring module path
6. Run `uv run pytest` and `uv run ruff check src/ tests/` before committing
