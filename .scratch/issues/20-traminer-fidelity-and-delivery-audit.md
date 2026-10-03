# TraMineR fidelity and delivery audit: wrong formulas, impersonating metrics, no oracle tests, no CI

**Status:** `resolved` (dev, 2026-09-29)
**Priority:** high — work this before any other open issue
**Type:** bug / process
**Source:** critical review 2026-09-28, full evidence in
[`.scratch/reviews/2026-09-28-critical-review.md`](../reviews/2026-09-28-critical-review.md)
**Related:** issue 01 (OM subcost bug, still `needs-info`), issue 15 (distance
engine seam — the threaded path lives there), issue 16 (quality placement)
**Source files:** `src/yasqat/statistics/descriptive.py`,
`src/yasqat/metrics/om_variants.py`, `src/yasqat/metrics/optimal_matching.py`,
`src/yasqat/core/pool.py`, `src/yasqat/core/alphabet.py`,
`.github/workflows/`, `pyproject.toml`, `docs/why-yasqat.md`

## Description

The house rule says TraMineR is the oracle for correctness, naming, and scope.
Nothing enforces it. A spot check on four toy sequences found two headline
statistics computing the wrong formula and three OM variants that carry
TraMineR names but not TraMineR semantics. No test in the tree pins a
TraMineR-derived value; the affected tests assert only `> 0.0`. Separately,
no CI runs tests, lint, or types on push or PR, and mypy does not pass.

| Sequence | turbulence (yasqat) | seqST (formula) | complexity (yasqat) | seqici (formula) |
|---|---|---|---|---|
| AABB | 0.00 | 3.00 | 0.354 | 0.408 |
| ABAB | 2.00 | 3.58 | 0.612 | 0.707 |
| AAAA | 0.00 | 1.00 | 0.000 | 0.000 |
| ABCD | 2.00 | 4.00 | 0.866 | 1.000 |

Alphabet {A,B,C,D}. Reference values derived from TraMineR's published
formulas; TraMineR is not installed on the dev machine.

## Defects

1. **`turbulence`** uses spell count where Elzinga's formula uses phi (number
   of distinct subsequences of the DSS), divides by mean duration, and lacks
   the max-variance term. Reference:
   `T = log2(phi * (s2max + 1) / (s2 + 1))`, `s2max = (n - 1) * (1 - tbar)^2`.
   `subsequence_count` already computes phi. Check `normalized_turbulence`
   against TraMineR `seqST(norm=TRUE)` once the base is fixed.
2. **`complexity_index`** computes `sqrt(transitions * distinct_states) / len`.
   TraMineR `seqici` is `sqrt((transitions / (len - 1)) * (H / log(|A|)))`.
3. **`omloc_distance`** weights substitution by distance from the sequence
   boundary. TraMineR OMloc (Hollister 2009) uses context-dependent costs
   from the neighbouring states with `expcost` / `context` parameters.
4. **`omspell_distance`** scales substitution by
   `1 / sqrt(spell_len_a * spell_len_b)`. TraMineR OMspell runs OM over spell
   sequences with duration-sensitive costs and `expcost`.
5. **`omstran_distance`** does not follow TraMineR's transition-sequence
   construction with the `otto` origin/transition weight.
6. **`optimal_matching_distance(normalize=True)`** docstring says "maximum
   possible distance"; the code divides by max length (TraMineR
   `maxlength`, which is fine). Fix the docstring, not the code.
7. **Null state** in the state column raises a bare `TypeError` from the
   alphabet sort instead of a `ValueError` at the public boundary.
8. **`n_jobs > 1`** submits one future per pair; at n=1000, T=24 it is 6×
   slower than sequential (5.5 s vs 0.9 s) and only breaks even at T≈100.
   The unreleased changelog entry advertises it as a win.

## Delivery gaps

- No workflow runs `pytest`, `ruff`, or `mypy` on push/PR; only the
  tag-triggered publish job runs tests.
- `mypy src/yasqat/` aborts on numpy stubs under the project's
  `python_version = 3.11` in the Python 3.14 venv; with `--python-version
  3.12` it reports 4 errors in `src/yasqat/clustering/pam.py`.
- `uv.lock` is gitignored; classifiers stop at 3.13 while dev runs 3.14.
- `pyarrow` is a hard dependency nothing imports.
- `docs/why-yasqat.md` claims "the test suite pins expected values against
  [TraMineR]" and "ruff and mypy run on every change". Neither is true.
- Guide outputs are pasted, not executed; docstring `>>>` examples are not
  doctested.

## Tasks (in order)

- [x] **Oracle = TraMineR source, not an R run.** The CRAN source is public
      (`github.com/cran/TraMineR`, 2.2-14). Definitions are read from
      `R/*.R` and `src/*.cpp` and reference values derived by hand, as
      issue 19 did for WeightedCluster. Done for seqST, seqivardur,
      seqsubsn, seqici, seqtransn, seqient. Still to read for the metrics:
      `OMdistance.cpp`, `OMPerdistance*.cpp` (OMspell, `timecost`),
      `OMVIdistance.cpp` (OMloc, `localcost`/`timecost`), `seqdist-OMstran.R`.
- [x] Fix `turbulence` and `normalized_turbulence` to the reference formula.
- [x] Fix `complexity_index` to `seqici`.
- [x] Rename `omloc`, `omspell`, `omstran` (user decision 2026-09-29) to
      `om_boundary_weighted_distance` / `om_spell_scaled_distance` /
      `om_transition_penalty_distance`, keys `om_boundary` /
      `om_spellscaled` / `om_transpenalty`.
- [x] Implement TraMineR's real OMloc, OMspell, OMstran from the C++/R
      source: `metrics/omloc.py`, `metrics/omspell.py`, `metrics/omstran.py`
      (+ `metrics/_normalize.py` for `normalizeDistance`), under the
      TraMineR names and dispatch keys. OMstran covers TraMineR's defaults
      only (`previous=FALSE`, `add.column=TRUE`).
- [x] Fix the OM `normalize` docstring.
- [x] Validate null states with a `ValueError` at the `SequencePool` /
      `StateSequence` boundary.
- [x] Either fix the threaded distance path (chunk pairs, or move the pair
      loop into numba `prange`) or make the docstring and changelog say it
      only pays off for long sequences. Re-run the n=1000 T=24 benchmark.
- [x] Replace `assert val > 0.0` in the turbulence/complexity/OM-variant
      tests with pinned values.
- [x] Add `.github/workflows/ci.yml`: pytest + ruff check + ruff format
      --check + mypy on push and PR, on a 3.11–3.13 matrix.
- [x] Make mypy pass (fix `pam.py`, set `python_version` so numpy stubs
      parse); commit `uv.lock`; drop `pyarrow` from dependencies.
- [x] Rewrite the "Fidelity over novelty" and "Engineering discipline"
      paragraphs of `docs/why-yasqat.md` to state only what the repo proves.
- [x] Re-run the docs guide examples and paste the corrected outputs (or
      wire up doctest / myst-nb so they cannot drift again).
- [x] CHANGELOG entries under Unreleased (Fixed, Changed).

## Out of scope here (file separately if wanted)

Scope questions from the review: dropping DTW/SoftDTW/TWED and the financial
generator, adding case weights, missing-state semantics, multichannel,
reference-sequence distances, distinct-sequence dedup, and the doubled
class/function clustering API (issue 16).

## Comments

- 2026-09-28: Filed from the critical review. Review kept as the evidence
  record; this issue is the actionable list.
- 2026-09-29: Fixed turbulence, normalized_turbulence, complexity_index, the
  OM normalize docstring, null-state validation, the chunked `n_jobs` path,
  mypy, CI workflow, lockfile, pyarrow removal, docs claims, guide values,
  changelog. Correction to the table: ABAB's DSS is `ABAB` (phi = 12), so
  seqST = log2(12) = 3.585, not 3.00; the implementation returns 3.585.
  Open decisions for the user: (1) the OM variants keep their names with
  honest docstrings — reimplement to TraMineR or rename? (2) turbulence uses
  the population variance (ddof=0), consistent with the `s2_max` bound;
  TraMineR's default `type=1` may use the sample variance — settle with the
  oracle run. (3) `normalized_turbulence` divides by `T_max = length`
  per sequence; TraMineR normalises by the max over the pool's longest
  sequence, identical for equal-length pools. (4) `subsequence_count` keeps
  its non-DSS, empty-excluded default; TraMineR's `seqsubsn` default is
  DSS=TRUE including the empty subsequence. Still open: the oracle fixture
  (needs R + TraMineR), the OM-variant decision, the OM-variant tests.
- 2026-09-29 (later): User decisions: (1) rename, done; (2)–(4) follow
  TraMineR. Read from source: `seqST` type=1 uses the population variance
  and `(n-1)(1-tbar)^2`, which is what was implemented — no change;
  `seqST(norm=TRUE)` rescales `(T-1)/(maxT-1)` against a max-length
  sequence cycling the alphabet — implemented; `seqsubsn` defaults to
  DSS=TRUE and counts the empty subsequence — now the default. TanaT has no
  turbulence, complexity, or OM-variant code, so it is not the source of
  the old formulas. Remaining: the real OMloc/OMspell/OMstran, and pinned
  tests for the renamed heuristics (currently identity/symmetry only).
- 2026-09-29 (close): Implemented OMloc/OMspell/OMstran from the TraMineR
  source under the TraMineR names; the heuristics keep their new names
  and now have pinned tests. Also found and fixed on the way: plain OM's
  `normalize=True` divided by length instead of `length * indel`
  (TraMineR `maxlength`). Every task is done; resolving. Follow-ups worth
  their own issues: OMstran `previous=TRUE`, OMslen, and a
  TraMineR-style `norm=` option on the other OM-family metrics.
