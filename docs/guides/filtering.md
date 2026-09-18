# Conditioning on a situation

Most applied questions have the shape "given this situation, what happens?"
A situation is a subset of sequences chosen by where they start, what they
contain, how long they are, or when they were observed. In yasqat a situation
is just another pool, so everything that works on the population works on
the situation.

## Criteria

Six criteria live in `yasqat.filters`. Each answers one question about a
sequence.

| Criterion | Keeps sequences that… |
|---|---|
| `LengthCriterion(min_length, max_length, exact_length)` | have a given number of observations |
| `TimeCriterion(start_after, start_before, end_after, end_before, contains_time)` | start, end, or cover a given time |
| `ContainsStateCriterion(states, require_all, exclude)` | contain any (or all) of some states, or exclude them |
| `StartsWithCriterion(states)` | begin with a given prefix |
| `QueryCriterion(expression)` | satisfy an arbitrary polars expression over the frame |
| `SequenceCriterion` | the base class, subclass it for anything else |

## `pool.filter`

```python
from yasqat.filters import ContainsStateCriterion, StartsWithCriterion

situation = pool.filter(
    [
        StartsWithCriterion(states=["unemployed"]),
        ContainsStateCriterion(states=["training"]),
    ],
    combine="and",
)
len(situation)       # 83 of the 300
```

`combine="and"` keeps sequences matching every criterion; `"or"` keeps those
matching any. The result is a `SequencePool` that keeps the parent's column
config and its **full alphabet**, so encodings stay comparable with the
parent and a state that no longer occurs is still a known state.

`filter_sequences(pool, criteria)` does the same selection but returns the
matching rows as a DataFrame, for when you want the rows rather than a
container.

## Filter, then score

Because the situation is a pool, the rest of the pipeline needs no
adaptation.

```python
from yasqat.statistics import association_rules, state_distribution
from yasqat.clustering import pam_clustering

# What follows for people who started unemployed and went through training?
association_rules(situation, min_support=0.3, min_confidence=0.6, max_length=3)

# How does their time budget differ from the population's?
state_distribution(situation)

# Do they form their own sub-typology?
dm = situation.compute_distances(method="om")
pam_clustering(dm, n_clusters=2)
```

Comparing a rule's lift inside a situation with its lift on the whole
population is the single most informative move in this style of analysis:
it separates "this pattern is common" from "this pattern is characteristic
of these people". The [mining guide](mining.md#given-a-situation) shows the
two numbers diverging on the subscription accounts.

## Situations you did not know how to write

Sometimes the interesting situation is not a filter you can state; it is a
cluster you discover. The workflow is the same in reverse: cluster the
population, then build one pool per cluster and score each.

```python
result = pam_clustering(pool.compute_distances(method="om"), n_clusters=3, random_state=0)
for cluster in range(result.n_clusters):
    ids = result.get_cluster_members(cluster)
    from yasqat.filters import QueryCriterion
    members = pool.filter(QueryCriterion(pl.col("id").is_in(ids)))
    print(cluster, len(members), association_rules(members, min_support=0.5).height)
```

Rules that are strong in one cluster and absent in the others are the
cluster's signature, and usually the most useful output of the whole
exercise.

## Cheaper subsets

Two pool methods cover the common cases without building criteria:

```python
pool.filter_by_length(min_length=12)   # drop short histories
pool.sample(500, seed=0)               # a random subset, e.g. before a large distance matrix
```
