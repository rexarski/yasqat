"""Partition quality indices over a distance matrix.

These read only a pairwise :class:`DistanceMatrix` (or raw array) and a
labelling, so they live next to the matrix they measure rather than next to
any clustering algorithm. WeightedCluster's ``wcClusterQuality`` is the
reference for PBC, HG and R2.
"""

from __future__ import annotations

import numpy as np

from yasqat.metrics.base import DistanceMatrix


def silhouette_scores(
    dist_matrix: DistanceMatrix | np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute per-point silhouette scores.

    For each point i:
        a(i) = mean distance to other points in same cluster
        b(i) = min over other clusters of mean distance to that cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Cluster labels, integer array of length n.

    Returns:
        Array of silhouette scores, one per point, in [-1, 1].
    """
    dist_matrix = DistanceMatrix.coerce(dist_matrix).values
    n = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    scores = np.zeros(n, dtype=np.float64)

    if n_clusters <= 1 or n_clusters >= n:
        return scores

    for i in range(n):
        cluster_i = labels[i]

        # a(i): mean distance to points in same cluster
        same_mask = labels == cluster_i
        same_count = int(np.sum(same_mask)) - 1  # exclude self
        if same_count > 0:
            a_i = float(np.sum(dist_matrix[i][same_mask])) / same_count
        else:
            a_i = 0.0

        # b(i): min mean distance to any other cluster
        b_i = float("inf")
        for label in unique_labels:
            if label == cluster_i:
                continue
            other_mask = labels == label
            other_count = int(np.sum(other_mask))
            if other_count > 0:
                mean_dist = float(np.sum(dist_matrix[i][other_mask])) / other_count
                if mean_dist < b_i:
                    b_i = mean_dist

        # Silhouette score
        denom = max(a_i, b_i)
        if denom > 0:
            scores[i] = (b_i - a_i) / denom
        else:
            scores[i] = 0.0

    return scores


def silhouette_score(
    dist_matrix: DistanceMatrix | np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the mean silhouette score (Average Silhouette Width).

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Cluster labels, integer array of length n.

    Returns:
        Mean silhouette score in [-1, 1]. Higher is better.
    """
    scores = silhouette_scores(dist_matrix, labels)
    return float(np.mean(scores))


def cluster_quality(
    dist_matrix: DistanceMatrix | np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute multiple cluster quality metrics.

    Metrics (definitions follow WeightedCluster's ``wcClusterQuality``, the
    TraMineR-side reference; all are "higher is better"):

        - ASW: Average Silhouette Width (mean silhouette score).
        - PBC: Point Biserial Correlation, ``-corr(distance, same-cluster)``
          over every cell of the full distance matrix, diagonal included.
          Positive when same-cluster pairs are closer than different-cluster
          pairs.
        - HG: Hubert's Gamma, the Goodman-Kruskal gamma between distance and
          the different-cluster indicator: ``(C - D) / (C + D)`` where a
          concordant pair of cells has the larger distance in the
          different-cluster cell. Pairs tied on distance are ignored.
        - R2: Proportion of variance explained by clustering, computed on
          squared distances (``1 - within SS / total SS``). This is
          WeightedCluster's ``R2sq``; its ``R2`` uses raw distances.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Cluster labels, integer array of length n.

    Returns:
        Dictionary with keys "ASW", "PBC", "HG", "R2".
    """
    dist_matrix = DistanceMatrix.coerce(dist_matrix).values
    n = len(labels)
    unique_labels = np.unique(labels)

    # ASW
    asw = silhouette_score(dist_matrix, labels)

    # PBC and HG are defined over every cell of the full matrix, diagonal
    # included, which is how the reference weights the pairs (each
    # off-diagonal pair twice, each diagonal zero-distance cell once).
    cells_dist = dist_matrix.astype(np.float64).ravel()
    cells_same = (labels[:, None] == labels[None, :]).astype(np.float64).ravel()

    pbc = -_pearson_correlation(cells_dist, cells_same)
    hg = _goodman_kruskal_gamma(cells_dist, cells_same)

    # R2: 1 - (within-cluster SS / total SS)
    # Total SS = sum of squared distances from each point to overall centroid
    # In distance matrix terms:
    # total_ss = sum of all squared distances / (2*n)
    # within_ss = sum of squared within-cluster distances / (2*n_k) for each cluster
    total_sq = float(np.sum(dist_matrix**2)) / (2 * n)

    within_sq = 0.0
    for label in unique_labels:
        mask = labels == label
        n_k = int(np.sum(mask))
        if n_k > 1:
            cluster_dists = dist_matrix[np.ix_(mask, mask)]
            within_sq += float(np.sum(cluster_dists**2)) / (2 * n_k)

    r2 = 1.0 - within_sq / total_sq if total_sq > 0 else 0.0

    return {
        "ASW": asw,
        "PBC": pbc,
        "HG": hg,
        "R2": r2,
    }


def distance_to_center(
    dist_matrix: DistanceMatrix | np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute mean distance from each point to its cluster center.

    The "center" is defined as the mean distance to all other points
    in the same cluster (since we work with distances, not coordinates).

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Cluster labels, integer array of length n.

    Returns:
        Array of mean within-cluster distances, one per point.
    """
    dist_matrix = DistanceMatrix.coerce(dist_matrix).values
    n = len(labels)
    distances = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mask = labels == labels[i]
        n_k = int(np.sum(mask))
        if n_k > 1:
            distances[i] = float(np.sum(dist_matrix[i][mask])) / (n_k - 1)

    return distances


def _goodman_kruskal_gamma(dist: np.ndarray, same: np.ndarray) -> float:
    """Goodman-Kruskal gamma between distance and the different-cluster flag.

    A pair of cells is concordant when the cell with the larger distance is
    the different-cluster one, discordant when it is the same-cluster one.
    Pairs tied on distance, or with the same flag, do not count. This mirrors
    the distinct-distance sweep in WeightedCluster's ``clusterqualitybody.cpp``
    (each ordered pair of cells is visited from both sides, which scales
    concordant and discordant counts equally and leaves the ratio intact).
    """
    order = np.argsort(dist, kind="stable")
    dist = dist[order]
    same = same[order]
    _, first = np.unique(dist, return_index=True)
    n_at = np.diff(np.append(first, len(dist))).astype(np.float64)
    s0 = np.add.reduceat(same, first)  # weight of same-cluster cells per distance
    s1 = n_at - s0  # weight of different-cluster cells per distance
    tot0, tot1 = float(s0.sum()), float(s1.sum())
    below0 = np.cumsum(s0) - s0  # strictly smaller distances, same cluster
    below1 = np.cumsum(s1) - s1  # strictly smaller distances, different cluster
    above0 = tot0 - below0 - s0
    above1 = tot1 - below1 - s1
    concordant = float((s1 * below0).sum() + (s0 * above1).sum())
    discordant = float((s0 * below1).sum() + (s1 * above0).sum())
    if concordant + discordant == 0:
        return 0.0
    return (concordant - discordant) / (concordant + discordant)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    dx = x - mean_x
    dy = y - mean_y

    num = float(np.sum(dx * dy))
    denom = float(np.sqrt(np.sum(dx**2) * np.sum(dy**2)))

    if denom == 0:
        return 0.0

    return num / denom
