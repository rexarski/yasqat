# Guides

Each guide is a runnable walk through one stage of the pipeline, with the
output you should see. The examples use two datasets that ship with the
package's synthetic generators or fit in ten lines: a three-state
labour-market population generated from a Markov chain, and ten hand-written
subscription accounts small enough to check the numbers by hand.

- [Loading data](loading-data.md): from files, DataFrames, or intervals to a
  `SequencePool`.
- [Measuring distance](distances.md): optimal matching worked by hand, the
  metric families, substitution costs, and how to choose.
- [Clustering](clustering.md): PAM, CLARA, and hierarchical typologies,
  choosing `k`, reading a cluster through its medoid.
- [Describing sequences](statistics.md): transitions, time in state, entropy
  and turbulence, normative indicators, discrepancy analysis.
- [Mining patterns](mining.md): frequent subsequences and sequential
  association rules, with every measure derived on paper.
- [Conditioning on a situation](filtering.md): filter criteria, `pool.filter`,
  and the filter-then-score chain.

```{toctree}
:maxdepth: 1
:hidden:

loading-data
distances
clustering
statistics
mining
filtering
```

## The two example datasets

Both guides' datasets are built like this; later pages assume `pool` and
`subs` exist.

```python
import numpy as np
import polars as pl
from yasqat.core import SequencePool
from yasqat.synthetic import generate_markov_sequences

# 300 people observed monthly for two years, three labour-market states.
T = np.array(
    [[0.85, 0.10, 0.05],   # employed   -> employed, unemployed, training
     [0.30, 0.55, 0.15],   # unemployed -> ...
     [0.40, 0.10, 0.50]]   # training   -> ...
)
df = generate_markov_sequences(
    n_sequences=300, sequence_length=24, transition_matrix=T,
    states=["employed", "unemployed", "training"], seed=42,
)
pool = SequencePool(df)

# Ten subscription accounts, written out so every number can be checked.
accounts = {
    1: "free free free paid paid paid",     2: "free free support paid paid paid",
    3: "free support support churn",        4: "free free paid paid paid paid",
    5: "free support paid paid churn",      6: "free free free free churn",
    7: "free support support paid paid",    8: "free paid paid paid paid paid",
    9: "free free support churn",           10: "free support paid paid paid paid",
}
rows = [(i, t, s) for i, seq in accounts.items() for t, s in enumerate(seq.split())]
subs = SequencePool(pl.DataFrame(rows, schema=["id", "time", "state"], orient="row"))
```

```python
pool.describe()
# {'n_sequences': 300, 'n_states': 3, 'states': ['employed', 'training', 'unemployed'],
#  'total_observations': 7200, 'min_length': 24, 'max_length': 24, ...}
subs.describe()
# {'n_sequences': 10, 'n_states': 4, 'states': ['churn', 'free', 'paid', 'support'],
#  'total_observations': 53, 'min_length': 4, 'max_length': 6, ...}
```

Note that the alphabet is sorted alphabetically when inferred, so the encoded
index of `employed` is 0, `training` is 1, and `unemployed` is 2 regardless of
the order you listed them. Pass an explicit `Alphabet` if you need a different
order, for instance to make a substitution matrix easier to read.
