# CLAUDE.md — yasqat development guide

Guidance for working *on* yasqat. This is not package documentation — for what a
function does or what it returns, read its docstring or `CONTEXT.md`. Keep this
file about non-obvious house rules and decisions that the code can't tell you.

## What yasqat is

**yasqat** (Yet Another Sequence Analytics Toolkit) — a polars-native library for
categorical sequence analysis (labour-market, health, education trajectories).
Python ≥ 3.11, hatchling build, `src/yasqat/` layout, tests in `tests/` mirroring
`src/`.

**Reference implementations:** **TraMineR** (R) is the oracle for algorithm
correctness, naming, and feature scope — when in doubt, match its behaviour.
**TanaT** (Python) is a secondary reference for Python-specific adaptations.

## Architecture

```
src/yasqat/
├── core/        # Alphabet, StateSequence, SequencePool
├── metrics/     # Pairwise distance metrics → DistanceMatrix
├── statistics/  # Transition rates, descriptive/normative stats, mining, trees
├── clustering/  # PAM, CLARA, hierarchical, quality indices
├── filters/     # Sequence filtering criteria
├── io/          # Loaders (CSV/parquet/JSON/DataFrame → SequencePool)
└── synthetic/   # Markov / financial generators
```

**Data flow:** raw data → `io` loaders → `SequencePool` (over a shared `Alphabet`)
→ `metrics` → `DistanceMatrix` → `clustering` / `statistics`. Every public method
returns a polars `DataFrame`; users bring their own plotting tool.

## Conventions

**Style** — ruff (line-length 88, py311 target); mypy strict (relaxed for
`metrics.*` because numba decorators are untyped). Every file opens with
`from __future__ import annotations`. Use the `TYPE_CHECKING` guard for
hint-only imports. Prefer `@dataclass`(`frozen=True` for value types).

**Naming** — modules `snake_case.py`, one concept per file; classes `PascalCase`;
functions `snake_case`. Control public API with `__all__` in each `__init__.py`.

**Metrics** — a metric is a free function `name_distance(seq_a, seq_b, **kwargs)
-> float` over integer-encoded arrays (use `@numba.njit` for the inner loop).
Register it in the dispatch dict in `SequencePool.compute_distances`
(`core/pool.py`) — that is the single seam for pairwise/matrix computation.
`DistanceMatrix` and `build_substitution_matrix` live in `metrics/base.py`.

**Dependencies** — polars for DataFrames (never pandas in core code; pyarrow only
as an interop bridge), numpy for numeric arrays, numba `@njit` for hot inner loops
in metrics, scipy for hierarchical linkage.

**Errors** — raise `ValueError` with a clear message for bad input; validate at
public API boundaries and trust internal calls; never `assert` in library code.

**Sequence containers** — `statistics.*` and `filters.*` functions accept
*either* a `StateSequence` or a `SequencePool`. Type the argument as
`core.protocols.SequenceData` (the shared
`data`/`config`/`alphabet`/`sequence_ids` surface) and
normalize at the top of the body with `SequencePool.coerce(sequence)` (or
`StateSequence.coerce(sequence)` when you need the format-conversion methods).
Don't reintroduce `StateSequence | SequencePool` unions or inline
`isinstance`/`_get_pool` coercion — `coerce` is the one seam.

**Testing** — pytest with `--strict-markers`; test paths mirror source
(`tests/test_metrics/test_hamming.py` ↔ `src/yasqat/metrics/hamming.py`); classes
`TestFeatureName` with descriptive methods, all annotated `-> None`; shared
fixtures in `tests/conftest.py`; `pytest.approx` / `np.testing.assert_allclose`
for floats; `pytest.raises` for exceptions. **Pin expected values** — never
`assert result > 0`.

## Commands

```bash
uv run pytest                                   # all tests
uv run pytest --cov=yasqat --cov-report=term-missing
uv run ruff check src/ tests/                   # lint
uv run ruff format src/ tests/                  # format
uv run mypy src/yasqat/                         # type check
```

## Adding a feature

1. Find the corresponding TraMineR/TanaT function for reference.
2. Create a `snake_case.py` module in the right subpackage; type hints + docstrings.
3. Export from the subpackage `__init__.py` (`__all__`).
4. Write tests mirroring the module path.
5. Run `uv run pytest` and `uv run ruff check src/ tests/` before committing.

## Agent skills

- **Issue tracker** — issues/PRDs are local markdown under `.scratch/<feature-slug>/`;
  external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.
- **Triage labels** — five canonical roles recorded as `Status:` lines in each
  issue file (not GitHub labels). See `docs/agents/triage-labels.md`.
- **Domain docs** — single-context: `CONTEXT.md` + `docs/adr/` at the repo root.
  See `docs/agents/domain.md`.
