# yasqat

**Yet Another Sequence Analytics Toolkit**

[![PyPI](https://img.shields.io/pypi/v/yasqat)](https://pypi.org/project/yasqat/)
[![Python](https://img.shields.io/pypi/pyversions/yasqat)](https://pypi.org/project/yasqat/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A modern Python library for categorical sequence analysis, built on
[polars](https://pola.rs/) and [plotnine](https://plotnine.org/). Designed for
social-science and life-course research — labour-market trajectories, health
pathways, educational histories, and similar domains.

Inspired by [TraMineR](http://traminer.unige.ch/) (R) and
[TanaT](https://tanat.gitlabpages.inria.fr/core/tanat/) (Python).

## Features

- **Polars-native data structures** — `Alphabet`, `StateSequence`,
  `IntervalSequence`, `SequencePool` for fast sequence manipulation
- **Distance metrics** — Optimal Matching, Hamming, LCS, LCP, RLCP, DTW,
  SoftDTW, Chi², Euclidean, DHD, TWED, and OM variants (OMloc, OMspell,
  OMstran, NMS, NMSMST, SVRspell), with convenience length/similarity
  wrappers for LCS, LCP, and RLCP
- **Substitution costs** — constant, transition-rate, indels, indelslog,
  future (chi-squared), features (Gower distance)
- **Clustering** — PAM (k-medoids) with `.predict()`, CLARA, hierarchical
  (scipy linkage); parallel pairwise distance computation via `n_jobs`
- **Cluster quality** — silhouette (ASW), Point Biserial, Hubert's Gamma,
  R², PAM range analysis, distance to center, representative extraction
- **Discrepancy analysis** — pseudo-ANOVA with permutation tests,
  multi-factor discrepancy, dissimilarity trees
- **Descriptive statistics** — entropy, transition rates, complexity,
  turbulence, spell counts, visited states, modal states (with time
  granularity), sequence frequencies, log-probabilities, subsequence
  counts (with state filtering and log-transform)
- **Normative indicators** — volatility, precarity, insecurity, degradation,
  badness, integration (per-state), proportion positive
- **Subsequence mining** — frequent subsequence discovery with support
  thresholds and minimum length, returned as polars DataFrames
- **Visualization** — index, distribution, frequency, spell duration,
  timeline, modal state, mean time, parallel coordinate, sunburst, and
  tree plots (all return composable `ggplot` objects)
- **Filtering** — length, time, state, and pattern-based sequence filtering
- **Data I/O** — CSV, Parquet, and DataFrame loading (Hive/Spark/Arrow
  interop) with automatic type inference
- **Synthetic data** — Markov-chain and financial trajectory generators

## Installation

```bash
pip install yasqat
```

## Quick start

```python
from yasqat.io import load_csv

# Load sequences from CSV (also: load_dataframe, load_parquet)
pool = load_csv("trajectories.csv", id_col="id", time_col="time", state_col="state")

# Compute pairwise distances and cluster
dm = pool.compute_distances(method="om", indel=1.0, n_jobs=4)

from yasqat.clustering import pam_clustering
result = pam_clustering(dm, n_clusters=4)

# Descriptive statistics
from yasqat.statistics import longitudinal_entropy, turbulence
longitudinal_entropy(pool.state_sequence)
turbulence(pool.state_sequence)

# Visualize (all return composable ggplot objects)
from yasqat.visualization import index_plot, sunburst_plot
index_plot(pool)
sunburst_plot(pool)
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/rexarski/yasqat.git
cd yasqat
uv venv && source .venv/bin/activate  # or activate.fish
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/yasqat/
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [TraMineR](http://traminer.unige.ch/) (R) and
  [TanaT](https://gitlab.inria.fr/tanat/core/tanat) (Python)
- Built with [polars](https://pola.rs/), [plotnine](https://plotnine.org/),
  [numba](https://numba.pydata.org/), and [scipy](https://scipy.org/)
