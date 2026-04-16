"""Cluster quality metrics for sequence analysis.

Provides metrics to evaluate the quality of clustering results
based on pairwise distance matrices and cluster labels.
"""

from __future__ import annotations

import warnings

import numpy as np


def k_range(start: int, end: int) -> range:
    """Inclusive range of k values.

    ``k_range(2, 10)`` returns ``range(2, 11)``, i.e. all integers 2..10.
    This exists because Python's built-in ``range(2, 10)`` is *exclusive*
    on the upper bound, which is a common source of off-by-one confusion
    when calling :func:`pam_range`.

    Args:
        start: Smallest k (inclusive).
        end: Largest k (inclusive).

    Returns:
        A ``range`` object that iterates every integer from ``start``
        through ``end``, inclusive on both ends.

    Example:
        >>> list(k_range(2, 5))
        [2, 3, 4, 5]
    """
    return range(start, end + 1)


def silhouette_scores(
    dist_matrix: np.ndarray,
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
    dist_matrix: np.ndarray,
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
    dist_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute multiple cluster quality metrics.

    Metrics:
        - ASW: Average Silhouette Width (mean silhouette score)
        - PBC: Point Biserial Correlation between distances and
          cluster membership (0/1 same-cluster indicator)
        - HG: Hubert's Gamma (correlation between distances and
          binary same/different cluster indicator)
        - R2: Proportion of variance explained by clustering
          (1 - within-cluster SS / total SS)

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        labels: Cluster labels, integer array of length n.

    Returns:
        Dictionary with keys "ASW", "PBC", "HG", "R2".
    """
    n = len(labels)
    unique_labels = np.unique(labels)

    # ASW
    asw = silhouette_score(dist_matrix, labels)

    # Extract upper triangle pairs for correlation metrics
    pairs_dist = []
    pairs_same = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_dist.append(dist_matrix[i, j])
            pairs_same.append(1.0 if labels[i] == labels[j] else 0.0)

    pairs_dist_arr = np.array(pairs_dist, dtype=np.float64)
    pairs_same_arr = np.array(pairs_same, dtype=np.float64)

    # PBC: Point Biserial Correlation
    # Correlation between distances and same-cluster indicator
    # We use same-cluster = 1, different = 0
    # Expect negative correlation (same cluster -> small distance)
    pbc = _pearson_correlation(pairs_dist_arr, pairs_same_arr)

    # HG: Hubert's Gamma
    # Correlation between distances and different-cluster indicator
    pairs_diff_arr = 1.0 - pairs_same_arr
    hg = _pearson_correlation(pairs_dist_arr, pairs_diff_arr)

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
    dist_matrix: np.ndarray,
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
    n = len(labels)
    distances = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mask = labels == labels[i]
        n_k = int(np.sum(mask))
        if n_k > 1:
            distances[i] = float(np.sum(dist_matrix[i][mask])) / (n_k - 1)

    return distances


def pam_range(
    dist_matrix: np.ndarray,
    k_values: range | list[int] | tuple[int, int] | None = None,
    max_iter: int = 100,
    init: str = "build",
    *,
    k_range: range | list[int] | tuple[int, int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Run PAM clustering for a range of k values and return quality metrics.

    For each k in the range, runs PAM clustering and computes cluster
    quality metrics (ASW, PBC, HG, R2). This helps identify the optimal
    number of clusters.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        k_values: Iterable of k values to try. Accepts ``range``, ``list``,
            or a 2-tuple ``(start, end)`` which is treated as an inclusive
            range with a ``DeprecationWarning`` (use :func:`k_range` instead).
            If ``None``, uses ``k_range(2, min(n - 1, 10))`` — every integer
            k from 2 up to at most 10.
        max_iter: Maximum PAM iterations per k.
        init: PAM initialization method.

    Returns:
        Dictionary mapping k -> quality metrics dict.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering.quality import pam_range, k_range
        >>> dist = np.array([
        ...     [0, 1, 5, 6],
        ...     [1, 0, 5, 6],
        ...     [5, 5, 0, 1],
        ...     [6, 6, 1, 0],
        ... ], dtype=float)
        >>> results = pam_range(dist, k_values=k_range(2, 3))
        >>> sorted(results.keys())
        [2, 3]
    """
    from yasqat.clustering.pam import pam_clustering

    # Backwards-compat: accept legacy ``k_range=`` kwarg. The parameter was
    # renamed to ``k_values`` to free the name ``k_range`` for a dedicated
    # helper function (see :func:`k_range`).
    if k_range is not None:
        if k_values is not None:
            raise TypeError(
                "pam_range() received both 'k_values' and 'k_range'; "
                "use 'k_values' only (k_range= is a legacy alias)."
            )
        k_values = k_range

    # Unwrap DistanceMatrix to numpy array
    if hasattr(dist_matrix, "values") and not isinstance(dist_matrix, np.ndarray):
        dist_matrix = dist_matrix.values

    n = dist_matrix.shape[0]

    # Auto-expand a (start, end) 2-tuple into an inclusive range. This is a
    # common foot-gun: users write ``pam_range(dm, (2, 10))`` expecting k=2..10
    # but ``for k in (2, 10)`` only visits the endpoints.
    if isinstance(k_values, tuple) and len(k_values) == 2:
        warnings.warn(
            f"pam_range received a 2-tuple {k_values}; treating it as an "
            f"inclusive (start, end) range. Pass `k_range({k_values[0]}, "
            f"{k_values[1]})` explicitly to silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        k_values = range(k_values[0], k_values[1] + 1)

    if k_values is None:
        # Inclusive upper bound 10, but never exceed n-1. We can't call the
        # module-level ``k_range`` helper here because the keyword-only
        # parameter of the same name shadows it in this scope — spell the
        # inclusive range out explicitly.
        k_values = range(2, min(n - 1, 10) + 1)

    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        if k < 2 or k >= n:
            continue

        pam_result = pam_clustering(
            dist_matrix,
            n_clusters=k,
            max_iter=max_iter,
            init=init,
        )

        quality = cluster_quality(dist_matrix, pam_result.labels)
        quality["total_cost"] = pam_result.total_cost
        quality["n_iterations"] = float(pam_result.n_iterations)
        results[k] = quality

    return results


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
