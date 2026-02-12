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

## Quick Start

```python
import polars as pl
from yasqat.core import StateSequence, SequencePool, Alphabet
from yasqat.visualization import index_plot
from yasqat.synthetic import generate_financial_journeys

# Create a simple sequence
data = pl.DataFrame({
    "id": [1, 1, 1, 1, 2, 2, 2, 2],
    "time": [0, 1, 2, 3, 0, 1, 2, 3],
    "state": ["A", "A", "B", "C", "A", "B", "B", "C"],
})

seq = StateSequence(data)

# Calculate entropy
from yasqat.statistics import longitudinal_entropy, spell_count, sequence_log_probability
entropy = longitudinal_entropy(seq)
spells = spell_count(seq, per_sequence=True)
log_prob = sequence_log_probability(seq)

# Compute distance between sequences
from yasqat.metrics import optimal_matching
pool = SequencePool(data)
distances = pool.compute_distances(method="om")  # or "lcp", "rlcp", "euclidean", "chi2", "dtw", "twed", "omloc", "nms"

# Cluster sequences and evaluate quality
from yasqat.clustering import HierarchicalClustering, cluster_quality, clara_clustering
from yasqat.clustering import extract_representatives, pam_range
clusterer = HierarchicalClustering(n_clusters=2)
result = clusterer.fit(distances)
quality = cluster_quality(distances, result.labels)
print(result.labels, quality["ASW"])

# Extract representative sequences
reps = extract_representatives(distances, n_representatives=2, strategy="centrality")

# Run discrepancy analysis (pseudo-ANOVA)
from yasqat.statistics import discrepancy_analysis
disc = discrepancy_analysis(distances, result.labels, n_permutations=99)
print(f"pseudo-R2={disc.pseudo_r2:.3f}, p={disc.p_value:.3f}")

# Normative indicators
from yasqat.statistics import volatility, precarity, badness
vol = volatility(pool, positive_states={"A"}, negative_states={"C"})
prec = precarity(pool, negative_states={"C"})

# Recode states (merge A and B into one)
recoded_pool = pool.recode_states({"A": "AB", "B": "AB"})

# Convert between sequence types
event_seq = seq.to_event_sequence()
interval_seq = seq.to_interval_sequence()

# Filter sequences
from yasqat.filters import LengthCriterion
criterion = LengthCriterion(min_length=3)
filtered = criterion.filter(seq)

# Save/load data
from yasqat.io import save_parquet, load_parquet
save_parquet(seq, "sequences.parquet")

# Generate synthetic financial data
journeys = generate_financial_journeys(n_users=1000)

# Visualize
plot = index_plot(pool)
plot.save("sequences.png")
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

## Running the demo script

A comprehensive demo script is included at `test_run.py`:

```python
%run ./test_run
```

## Concepts

### Sequence Representations

| Format | Description | Example |
|--------|-------------|---------|
| **STS** (State-Time Sequence) | One state per time unit | `A,A,B,B,C,A` |
| **SPS** (State-Permanence) | Run-length encoded | `(A,2)-(B,2)-(C,1)-(A,1)` |
| **DSS** (Distinct Successive) | Duplicates removed | `A,B,C,A` |
| **Interval** | Start/end time per state | `(A, 0-5), (B, 5-10)` |

### Distance Metrics

| Method | Description | Use Case |
|--------|-------------|----------|
| Optimal Matching | Edit distance with indel/substitution | Variable-length sequences |
| Hamming | Position-wise mismatch count | Fixed-length sequences |
| DHD | Dynamic Hamming (position-dependent costs) | Fixed-length, time-varying contexts |
| LCS | Longest Common Subsequence | Order-focused comparison |
| DTW | Dynamic Time Warping | Variable timing, elastic matching |
| SoftDTW | Differentiable DTW | ML-compatible distance |
| LCP | Longest Common Prefix | Prefix-focused comparison |
| RLCP | Reverse Longest Common Prefix | Suffix-focused comparison |
| Chi2 | Chi-squared distribution | State proportion comparison |
| Euclidean | L2 on state proportion vectors | Distribution comparison |
| TWED | Time Warp Edit Distance | Temporal elasticity with stiffness |
| OMloc | Localized Optimal Matching | Position-dependent costs |
| OMspell | Spell-length sensitive OM | Spell-aware comparison |
| OMstran | Transition-sensitive OM | Transition-pattern comparison |
| NMS | Number of Matching Subsequences | Subsequence-based similarity |
| NMSMST | NMS with MST normalization | Normalized subsequence distance |
| SVRspell | Spell-based representativeness | Spell structure comparison |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [TanaT](https://gitlab.inria.fr/tanat/core/tanat) (Python) and [TraMineR](http://traminer.unige.ch/) (R)
- Built with [polars](https://pola.rs/), [plotnine](https://plotnine.org/), and [numba](https://numba.pydata.org/)
