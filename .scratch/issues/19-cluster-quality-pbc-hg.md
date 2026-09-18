# `cluster_quality`: PBC sign flipped, HG was not Hubert's Gamma

**Status:** `resolved` (dev, 2026-09-17)
**Type:** bug
**Source:** noticed while writing the clustering guide, 2026-09-17: PBC and HG
came back with equal magnitude and opposite sign (e.g. PBC -0.365, HG 0.365)
**Source file:** `src/yasqat/clustering/quality.py`

## Description

`cluster_quality` computed

- PBC as `corr(distance, same-cluster)` over the upper triangle, so a good
  partition scored *negative*;
- HG as `corr(distance, different-cluster)`, i.e. exactly `-PBC`, which is a
  Pearson correlation, not Hubert's Gamma.

## Reference

WeightedCluster `wcClusterQuality` (`src/clusterqualitybody.cpp`, the
TraMineR-side implementation):

- **PBC** (`ClusterQualHPG`) = `-1 * pearson(x = distance, y = same-cluster)`
  computed over the *full* matrix with the diagonal included (weight `w_i²`
  for cell `(i, i)`, `2·w_i·w_j` for each off-diagonal pair). Positive for a
  good partition.
- **HG** (`ClusterQualHG`) = Goodman-Kruskal gamma `(nc - nd) / (nc + nd)`
  between distance and the different-cluster indicator, over the same
  weighted cells, ignoring pairs tied on distance. Concordant = the larger
  distance belongs to the different-cluster cell.
- WeightedCluster's `R2` uses raw distances and `R2sq` squared distances;
  yasqat's `R2` is the squared-distance version and is now documented as
  such (unchanged).

## Resolution

Both measures re-implemented to the reference definitions over the full
matrix. The gamma uses the same distinct-distance sweep as the C++ code
(O(m log m) in the number of cells). Tests pin values from a brute-force
transcription of the reference on a clean two-cluster matrix (PBC 0.9874,
HG 1.0), an imperfect assignment (PBC 0.4100, HG 0.7037), and a 9-point
Euclidean matrix with three clusters (PBC 0.3947, HG 0.4294). No R
installation was available to run WeightedCluster itself; the brute force
follows its C++ line by line.
