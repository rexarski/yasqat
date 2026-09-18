# Describing sequences

These functions read the pool directly; no distance matrix is needed. Most
accept `per_sequence=True` to return one row per id instead of a population
aggregate. All results are polars DataFrames or plain floats.

## Where time goes

```python
from yasqat.statistics import state_distribution, mean_time_in_state

state_distribution(pool)
```

```
┌────────────┬───────┬────────────┐
│ state      ┆ count ┆ proportion │
╞════════════╪═══════╪════════════╡
│ employed   ┆ 4739  ┆ 0.658194   │
│ unemployed ┆ 1492  ┆ 0.207222   │
│ training   ┆ 969   ┆ 0.134583   │
└────────────┴───────┴────────────┘
```

Pass `time_point=0` to see the distribution at one step, which is how you
check that a cohort starts where you think it does. `per_sequence=True` gives
the same table per id, the input for a "share of time employed" covariate.

```python
mean_time_in_state(pool)
```

```
┌────────────┬────────────┬───────────┐
│ state      ┆ total_time ┆ mean_time │
╞════════════╪════════════╪═══════════╡
│ employed   ┆ 4739       ┆ 15.796667 │
│ training   ┆ 969        ┆ 3.23      │
│ unemployed ┆ 1492       ┆ 4.973333  │
└────────────┴────────────┴───────────┘
```

Over 24 months, the average person spends 15.8 in employment and 5.0
unemployed.

## How states flow

```python
from yasqat.statistics import transition_rates, transition_rate_matrix

transition_rates(pool)
```

```
┌────────────┬────────────┬───────┬──────────┐
│ from_state ┆ to_state   ┆ count ┆ rate     │
╞════════════╪════════════╪═══════╪══════════╡
│ employed   ┆ employed   ┆ 3871  ┆ 0.853019 │
│ employed   ┆ training   ┆ 221   ┆ 0.0487   │
│ employed   ┆ unemployed ┆ 446   ┆ 0.098281 │
│ …          ┆ …          ┆ …     ┆ …        │
│ unemployed ┆ employed   ┆ 404   ┆ 0.282517 │
│ unemployed ┆ training   ┆ 191   ┆ 0.133566 │
│ unemployed ┆ unemployed ┆ 835   ┆ 0.583916 │
└────────────┴────────────┴───────┴──────────┘
```

Each rate is the probability of the next state given the current one. This
population was generated from a Markov chain, and the recovered rates match
the generating matrix to within sampling noise (0.853 against 0.85 for staying
employed). `transition_rate_matrix` returns the same numbers as a numpy array
in alphabet order, and `exclude_self=True` drops the diagonal to look only at
moves.

## How eventful a path is

```python
from yasqat.statistics import longitudinal_entropy, turbulence, complexity_index, spell_count

longitudinal_entropy(pool)     # 0.692  (population mean, normalised to [0, 1])
turbulence(pool)               # 4.059
complexity_index(pool)         # 0.166
spell_count(pool)              # 6.82   spells per sequence on average
```

- **Longitudinal entropy** measures how evenly a sequence's time is spread
  across states. A whole career in one state scores 0; equal time in every
  state scores 1. It ignores order.
- **Turbulence** combines the number of distinct subsequences with the
  variance of spell durations, so it rewards both switching and irregular
  durations. Unlike entropy it does see order.
- **Complexity index** is a normalised blend of transitions and entropy in
  [0, 1].

With `per_sequence=True` each becomes a per-id column, ready to join to
covariates or to compare across clusters.

## Normative indicators

When states can be ranked as better or worse, a family of indicators measures
how a trajectory moves along that ranking. You declare the ranking; the
functions do not guess it.

```python
from yasqat.statistics import volatility, precarity

volatility(pool, positive_states={"employed"}, negative_states={"unemployed"})   # 0.123
precarity(pool, negative_states={"unemployed"})                                 # 0.198
```

Also available: `insecurity`, `degradation`, `badness`, `integration` (per
state), `proportion_positive`, and `objective_volatility`, a label-free variant
that needs no ranking. Their docstrings give the formulas and the TraMineR
function each corresponds to.

## Frequencies and typical sequences

```python
from yasqat.statistics import sequence_frequency_table, modal_states

sequence_frequency_table(pool, n_top=3)   # the most common whole sequences
modal_states(pool)                         # the most frequent state at each time step
```

```
┌──────┬─────────────┬───────────┬────────────┐
│ time ┆ modal_state ┆ frequency ┆ proportion │
╞══════╪═════════════╪═══════════╪════════════╡
│ 0    ┆ training    ┆ 109       ┆ 0.363333   │
│ 1    ┆ employed    ┆ 136       ┆ 0.453333   │
│ 2    ┆ employed    ┆ 172       ┆ 0.573333   │
```

The chain starts from a roughly uniform draw, and by month 2 employment is
already the modal state: the modal-state table is the cheapest picture of
convergence.

## Discrepancy analysis

Does a grouping explain the distances? Discrepancy analysis is a pseudo-ANOVA
on a distance matrix: it splits the total sum of squared distances into a
between-group and a within-group part and tests the ratio by permutation.

```python
from yasqat.statistics import discrepancy_analysis

labels = pam_clustering(dm, n_clusters=3, random_state=0).labels
disc = discrepancy_analysis(dm, labels, n_permutations=99, random_state=0)
disc
# DiscrepancyResult(pseudo_R2=0.2583, pseudo_F=51.7137, p=0.0100)
```

The grouping can be anything aligned to `dm.labels`: cluster membership, as
here, or a covariate such as gender or cohort. `multi_factor_discrepancy`
handles several covariates at once, and `dissimilarity_tree` grows a
regression tree on the distance matrix, splitting on the covariate that
explains the most discrepancy at each node.

The pseudo-R² of 0.258 says the three-cluster typology accounts for about a
quarter of the pairwise variation, and no permutation of the labels reached
the observed pseudo-F, hence `p = 0.01` at 99 permutations. That is the
sentence to put next to any typology you report.
