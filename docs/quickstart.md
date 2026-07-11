# Quick start

```python
from yasqat.io import load_csv

# Load sequences from CSV (also: load_dataframe, load_json, load_parquet).
# Every loader returns a SequencePool, the analysis container. Default column
# names are id/time/state; pass config=SequenceConfig(...) to override.
pool = load_csv("trajectories.csv")

# Compute pairwise distances and cluster
dm = pool.compute_distances(method="om", indel=1.0, n_jobs=4)

from yasqat.clustering import pam_clustering

result = pam_clustering(dm, n_clusters=4)

# Descriptive statistics
from yasqat.statistics import longitudinal_entropy, turbulence

longitudinal_entropy(pool)
turbulence(pool)

# Plot with your library of choice — yasqat methods return polars DataFrames
state_distribution = pool.to_state_sequence().state_per_sequence(proportion=True)
# Hand `state_distribution` to matplotlib, altair, etc.
```

## Mining frequent patterns and rules

```python
from yasqat.statistics import association_rules, frequent_subsequences

# Ordered patterns appearing in at least 20% of trajectories
patterns = frequent_subsequences(pool, min_support=0.2, max_length=4)

# Sequential association rules (antecedent => consequent) with confidence,
# lift, leverage and conviction
rules = association_rules(pool, min_support=0.2, min_confidence=0.6)
rules.filter(pl.col("lift") > 1.0)
```

See the [API reference](api/index.md) for the full public surface and the
[Glossary](glossary.md) for the domain vocabulary behind these functions.
