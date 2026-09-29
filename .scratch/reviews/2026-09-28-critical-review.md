# yasqat critical review — 2026-09-28

Scope: development process, feature scope, delivery. Audited `main` at
af08b58 (dev is identical; `demo-20250216` is a stale February branch;
`gh-pages` is the built site). All numbers below were measured in this
checkout on 2026-09-28. TraMineR values are derived from its published
formulas; TraMineR itself is not installed here, so no R run backs them.

## Verdict

yasqat is a well-organised codebase with a real correctness problem
underneath it. The engineering hygiene (coercion seams, changelog, ADRs,
613 passing tests in 9 s, ruff clean) is better than most solo projects.
But the project's own stated standard, "TraMineR is the oracle", is not
enforced anywhere, and several headline functions return numbers TraMineR
would not. The docs also make claims the repo does not back up. The package
is currently more polished than it is correct.

## 1. Correctness and fidelity

Hand-derived TraMineR formulas on four toy sequences over alphabet {A,B,C,D}:

| Sequence | turbulence (yasqat) | seqST (formula) | complexity (yasqat) | seqici (formula) |
|---|---|---|---|---|
| AABB | 0.00 | 3.00 | 0.354 | 0.408 |
| ABAB | 2.00 | 3.58 | 0.612 | 0.707 |
| AAAA | 0.00 | 1.00 | 0.000 | 0.000 |
| ABCD | 2.00 | 4.00 | 0.866 | 1.000 |

- **Turbulence uses the wrong formula.** Elzinga's T = log2(phi ×
  (s²max + 1) / (s² + 1)), where phi is the number of distinct subsequences
  of the DSS and s²max = (n−1)(1−t̄)². The docstring in
  `src/yasqat/statistics/descriptive.py` names phi correctly, but the code
  substitutes the spell count for phi, divides by mean duration, and has no
  max-variance term. The mismatch is structural and does not depend on
  whether the empty subsequence is counted or on the variance ddof
  convention. `subsequence_count` already computes phi and is not used.
- **Complexity index uses the wrong formula.** It computes
  sqrt(transitions × distinct states) / length. Elzinga's index (TraMineR
  `seqici`) is sqrt(transition rate × normalised entropy). Every value is
  off, and the docstring formula does not correspond to a published index.
- **OMloc, OMspell, OMstran are invented heuristics under TraMineR names.**
  TraMineR OMloc uses context-dependent substitution costs from the
  neighbouring states (Hollister 2009). yasqat weights substitution by
  distance from the sequence boundary. TraMineR OMspell runs OM over spell
  sequences with duration-sensitive costs and an `expcost` parameter. yasqat
  scales the cost by 1/sqrt(spell_len_a × spell_len_b). OMstran likewise
  does not follow TraMineR's transition-sequence construction. A user who
  picks these by name from Studer & Ritschard will get a different measure.
- **OM `normalize` docstring is wrong.** It says "normalize by the maximum
  possible distance"; the code divides by max length (which does match
  TraMineR's `maxlength` normalisation) and the test pins a normalised
  value of 2.0. Only the docstring needs fixing.
- **Tests cannot catch any of this.** `docs/why-yasqat.md` says the suite
  "pins expected values against" TraMineR. There is no TraMineR-derived
  value anywhere in `tests/`. The turbulence and complexity tests assert
  only `> 0.0`. The 17 OM-variant tests check identity, symmetry, and
  positivity and pin nothing non-trivial. Roughly 50 assertions compare to
  zero or a bound rather than a pinned value (a few p-value checks are
  legitimate). Coverage on the exotic metrics: DTW 34 %, SoftDTW 41 %, TWED
  41 %, OM variants 44 %.
- **The one known bug in the flagship metric is parked.** Issue 01 ("OM
  with a substitution matrix still has an error") has sat in `needs-info`
  since it was deferred from the 0.4.0 brainstorm in May, with no attempt
  to construct a repro.

What does match: longitudinal entropy reproduced the expected normalised
values in the same check, and the `trate` substitution cost formula
(2 − p_ij − p_ji) matches TraMineR's TRATE definition. The 0.5.0 fix that
matched PBC and Hubert's Gamma to WeightedCluster shows the oracle
discipline works when applied. It has not been applied broadly.

## 2. Scope and positioning

- **Breadth over depth.** Seventeen distance metrics, including DTW,
  SoftDTW, and TWED, which are time-series metrics with little standing in
  categorical sequence analysis. Meanwhile features every TraMineR user
  expects are absent: case weights, missing or censored states,
  multichannel sequences, distances to a reference sequence, and
  distinct-sequence deduplication before the pairwise loop. A null in the
  state column does not raise a `ValueError`; it crashes with a `TypeError`
  from the alphabet sort. A life-course researcher will hit one of these
  gaps in the first hour.
- **Positioning contradicts the code.** README and docs pitch social
  science and life-course research. The second-largest source file in the
  package (`synthetic/financial.py`, 588 lines) is a financial-institution
  journey generator with KYC and delinquency states, and the git history is
  full of journey-pathways decks. The actual driver looks like
  customer-journey analytics. Either audience is fine; the docs should
  describe the one the code serves.
- **Scale claims are inconsistent and partly untrue.** The why page says
  "thousands to a few hundred thousand sequences". The `compute_distances`
  docstring says to sample above about 500. Measured OM, sequential:

  | n | T | time |
  |---|---|---|
  | 500 | 24 | 0.23 s |
  | 1000 | 24 | 0.9 s |
  | 2000 | 24 | 3.6 s |
  | 1000 | 100 | 7.0 s |

  Sequential is fine at cohort scale. But the threaded path submits one
  future per pair, and `n_jobs=4` at n=1000 is 6× slower than sequential at
  length 24 (5.5 s vs 0.9 s) and only breaks even at length 100. Length 24
  is typical for this field, so the newly advertised parallelism is a net
  loss in the common case. At 100 000 sequences the dense matrix alone is
  80 GB; nothing in the design addresses that.
- **Public surface is wide and doubled.** 36 statistics functions, both
  class and function entry points for PAM and hierarchical clustering
  (issue 16, untriaged).

## 3. Delivery and process

- **No CI runs tests, lint, or types on push or PR.** Only the
  tag-triggered publish job runs pytest. The why page says ruff and mypy
  "run on every change". They do not.
- **mypy does not pass.** Under the project's own `python_version = 3.11`
  it aborts on numpy's stubs in the current Python 3.14 venv. Told to parse
  3.12 grammar, it reports four errors in `src/yasqat/clustering/pam.py`.
  Strict typing is a claim, not a gate.
- **Environment is not reproducible.** `uv.lock` is gitignored, classifiers
  stop at 3.13 while development runs on 3.14, and `pyarrow` is a hard
  dependency that nothing imports (`load_dataframe` accepts only polars),
  justified by a comment about pandas that the house rules forbid.
- **Docs drift is unguarded.** The guides' "real outputs" are pasted
  markdown, not executed (no myst-nb, no doctest). Docstring `>>>` examples
  are never run. The wrong turbulence values are therefore reproduced in the
  docs with authority.
- **Process is heavy for a project with no users.** All 12 PRs and 49
  issues are by one author, self-merged with no review or CI. GitHub has 0
  stars and 0 forks; PyPI shows 61 downloads in the last month, which is
  mirror noise. Yet the repo carries ADRs, a five-state triage vocabulary, a
  committed 319 KB primer PDF, and an in-repo markdown issue tracker. The
  GitHub issues were closed when the tracker moved, while `pyproject.toml`
  still points contributors at GitHub Issues.
- **Staleness.** A demo branch from February, a test file still named for
  the metric-class layer deleted in 0.5.0 (`test_metric_classes.py`), a
  behaviour-changing PBC sign fix sitting unreleased.

## Recommended order of work

1. Stop adding features until an oracle test layer exists. Install
   TraMineR, run mvad or biofam through both, and pin a fixture of real
   values per metric and statistic.
2. Fix turbulence and complexity. Reimplement OMloc/OMspell/OMstran to
   TraMineR's definitions, or rename them so they stop impersonating it.
3. Add a test-and-lint workflow on push and PR, commit the lockfile, make
   mypy pass.
4. Cut DTW, SoftDTW, TWED, and the financial generator, or move them to an
   extras package. Spend the freed attention on weights, missing states,
   and deduplication.
5. Rewrite the why page to say only what the repo can prove.
