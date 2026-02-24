# yasqat

**Yet Another Sequence Analytics Toolkit**

A modern Python library for sequence analysis, inspired by [TanaT](https://tanat.gitlabpages.inria.fr/core/tanat/) and [TraMineR](http://traminer.unige.ch/).

## Features

- **Polars-based data structures** for fast sequence manipulation
- **Multiple sequence types**: StateSequence, EventSequence, IntervalSequence with bidirectional conversion
- **Distance metrics**: Optimal Matching, Hamming, LCS, DTW, SoftDTW, LCP, RLCP, Chi2, Euclidean, DHD, TWED, OMloc, OMspell, OMstran, NMS, NMSMST, SVRspell
- **Substitution costs**: Constant, transition-rate, indels, indelslog, future (chi-squared), features (Gower distance)
- **Clustering**: Hierarchical clustering, PAM (k-medoids), CLARA (sampling-based PAM)
- **Cluster quality**: Silhouette scores (ASW), Point Biserial Correlation, Hubert's Gamma, R-squared, PAM range analysis, distance to center
- **Representative sequences**: Extract representatives by centrality, frequency, or density
- **Discrepancy analysis**: Pseudo-ANOVA (pseudo-F, pseudo-R2) with permutation tests, multi-factor ANOVA
- **Dissimilarity trees**: Recursive partitioning of distance matrices by covariates
- **State recoding**: Merge or rename states with automatic alphabet rebuild
- **Filtering**: Length, time, state-based, and pattern filtering
- **Data I/O**: CSV, JSON, Parquet support with polars
- **Trajectory**: Multi-sequence entity analysis
- **Descriptive statistics**: Entropy, transition rates, complexity, turbulence, normalized turbulence, spell counts, visited states, modal states, sequence frequencies, log-probabilities, subsequence count
- **Normative indicators**: Volatility, precarity, insecurity, degradation, badness, integration, proportion positive
- **Frequent subsequence mining**: Apriori-like discovery with support thresholds
- **Visualization**: Index plots, distribution plots, frequency plots, spell duration plots, timeline, modal state plots, mean time plots, parallel coordinate plots
- **Synthetic data generation**: Generate realistic user journey data

## Installation

### Using uv (recommended)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate.fish

# Install in development mode
uv pip install -e ".[dev]"
```

## Development

```bash
# Clone the repository
git clone https://github.com/rexarski/yasqat.git
cd yasqat

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src/yasqat

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [TanaT](https://gitlab.inria.fr/tanat/core/tanat) (Python) and [TraMineR](http://traminer.unige.ch/) (R)
- Built with [polars](https://pola.rs/), [plotnine](https://plotnine.org/), and [numba](https://numba.pydata.org/)
