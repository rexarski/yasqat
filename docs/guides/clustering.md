# Clustering

A typology is a partition of the distance matrix into groups of similar
trajectories, each summarised by a real sequence. This guide builds one from
the labour-market pool, chooses the number of clusters, and reads the result.

## PAM: partitioning around medoids

PAM (k-medoids) picks `k` actual sequences as centres and assigns every other
sequence to the nearest one. Because centres are real members, each cluster
has an interpretable exemplar.

```python
from yasqat.clustering import pam_clustering

dm = pool.compute_distances(method="om", indel=1.0)
result = pam_clustering(dm, n_clusters=3, random_state=0)

result.cluster_sizes()      # {0: 93, 1: 84, 2: 123}
result.get_medoid_ids()     # the three exemplar sequence ids
result.labels[:10]          # cluster index per sequence, aligned to dm.labels
result.to_dataframe().head()
```

```
┌─────┬─────────┬───────────┐
│ id  ┆ cluster ┆ is_medoid │
╞═════╪═════════╪═══════════╡
│ 0   ┆ 0       ┆ false     │
│ 1   ┆ 1       ┆ false     │
│ …   ┆ …       ┆ …         │
```

To see what a cluster *is*, look at its medoid and its state distribution:

```python
for cluster, medoid_id in enumerate(result.get_medoid_ids()):
    print(cluster, pool.get_sequence(medoid_id))
```

A `PAMClustering` class with `fit` and `predict` is also exported, for the
case where new sequences must be assigned to an existing typology.

## Choosing the number of clusters

`pam_range` fits several `k` and reports quality indices for each.

```python
from yasqat.clustering import pam_range

pam_range(dm, k_values=range(2, 6))
```

| k | ASW | R² | total cost |
|---|---|---|---|
| 2 | 0.190 | 0.182 | 3510 |
| 3 | 0.139 | 0.258 | 3272 |
| 4 | 0.122 | 0.289 | 3120 |
| 5 | 0.106 | 0.358 | 2944 |

Two indices pull in opposite directions, which is normal:

- **ASW**, average silhouette width, measures how much closer each sequence is
  to its own cluster than to the next one. Higher is better; here it prefers
  `k = 2`. Values below about 0.25 indicate weak structure, and this synthetic
  population is deliberately noisy.
- **R²** is the share of total discrepancy explained by the partition. It
  always rises with `k`, so look for the elbow rather than the maximum.

`cluster_quality(dm, labels)` returns the same indices for a single partition,
including Point Biserial Correlation (PBC) and Hubert's Gamma (HG). Reading the
table above, three clusters is the defensible choice: ASW does not collapse
and R² gains most of its value there.

## CLARA: PAM for large pools

CLARA runs PAM on several random samples and keeps the best set of medoids,
then assigns everyone. It is the route when the pool is too large for a full
distance matrix to be pleasant.

```python
from yasqat.clustering import clara_clustering

clara = clara_clustering(dm, n_clusters=3, n_samples=5, random_state=0)
clara.cluster_sizes()       # {0: 74, 1: 137, 2: 89}: a different, equally valid cut of the same matrix
```

## Hierarchical clustering

Agglomerative clustering over the same matrix, using scipy's linkage. Ward is
the default and usually the right one for OM distances.

```python
from yasqat.clustering import hierarchical_clustering

hier = hierarchical_clustering(dm, n_clusters=3, method="ward")
hier.linkage_matrix      # hand this to scipy.cluster.hierarchy.dendrogram
```

The result carries the full linkage matrix, so you can cut it at a different
height, or draw a dendrogram with scipy, without recomputing.

## Representatives

Beyond the medoid, `extract_representatives` selects a small set of sequences
that together cover a cluster: the most central ones by default, or the most
diverse.

```python
from yasqat.clustering import extract_representatives

reps = extract_representatives(dm.values, n_representatives=2, labels=result.labels)
reps.indices    # positions into dm.labels
reps.scores
```

## Is the typology real?

A partition always exists; the question is whether it explains anything.
[Discrepancy analysis](statistics.md#discrepancy-analysis) answers it with a
permutation test on the cluster labels: for the three-cluster solution above
it reports a pseudo-R² of 0.258 with `p = 0.01` over 99 permutations. Report
that number alongside the typology.
