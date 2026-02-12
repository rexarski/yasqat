"""CLARA (Clustering Large Applications) for sequences.

CLARA is a sampling-based extension of PAM for large datasets.
It repeatedly draws random subsamples, runs PAM on each, and
keeps the solution with the lowest total cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from yasqat.clustering.pam import _assign_clusters, _compute_cost, pam_clustering

if TYPE_CHECKING:
    from yasqat.metrics.base import DistanceMatrix


@dataclass
class CLARAClusteringResult:
    """Result of CLARA clustering."""

    labels: np.ndarray
    """Cluster labels for each sequence."""

    medoid_indices: np.ndarray
    """Indices of medoid sequences in the full distance matrix."""

    n_clusters: int
    """Number of clusters."""

    total_cost: float
    """Total cost of the best solution (on full data)."""

    n_samples: int
    """Number of random subsamples evaluated."""

    sample_size: int
    """Size of each random subsample."""

    best_sample_cost: float
    """Total cost of the best subsample solution."""

    def cluster_sizes(self) -> dict[int, int]:
        """Return the size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts, strict=True))


def clara_clustering(
    distance_matrix: DistanceMatrix | np.ndarray,
    n_clusters: int,
    n_samples: int = 5,
    sample_size: int | None = None,
    max_iter: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> CLARAClusteringResult:
    """
    Perform CLARA clustering (sampling-based PAM).

    CLARA repeatedly draws random subsamples from the data, runs PAM
    on each subsample, then evaluates each solution on the full dataset.
    The solution with the lowest total cost is returned.

    Args:
        distance_matrix: Pairwise distance matrix (DistanceMatrix or numpy array).
        n_clusters: Number of clusters to form.
        n_samples: Number of random subsamples to draw.
        sample_size: Size of each subsample. If None, uses
            min(40 + 2 * n_clusters, n) as in the original CLARA paper.
        max_iter: Maximum PAM iterations per subsample.
        random_state: Random state for reproducibility.

    Returns:
        CLARAClusteringResult with cluster labels and medoid information.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering.clara import clara_clustering
        >>> dist = np.random.default_rng(42).random((50, 50))
        >>> dist = (dist + dist.T) / 2
        >>> np.fill_diagonal(dist, 0)
        >>> result = clara_clustering(dist, n_clusters=3, random_state=42)
        >>> result.n_clusters
        3
    """
    # Extract numpy array from DistanceMatrix if needed
    if hasattr(distance_matrix, "matrix"):
        dist_array = distance_matrix.matrix
    else:
        dist_array = np.asarray(distance_matrix, dtype=np.float64)

    n = dist_array.shape[0]
    if dist_array.shape != (n, n):
        raise ValueError("Distance matrix must be square")

    if n_clusters > n:
        raise ValueError(
            f"Number of clusters ({n_clusters}) cannot exceed number of samples ({n})"
        )

    # Default sample size: min(40 + 2k, n) following Kaufman & Rousseeuw
    if sample_size is None:
        sample_size = min(40 + 2 * n_clusters, n)
    sample_size = min(sample_size, n)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    best_medoids = None
    best_cost = float("inf")
    best_sample_cost = float("inf")

    for _ in range(n_samples):
        # Draw random subsample
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample_idx.sort()

        # Extract sub-distance-matrix
        sub_dist = dist_array[np.ix_(sample_idx, sample_idx)]

        # Run PAM on subsample
        sub_result = pam_clustering(
            sub_dist,
            n_clusters=n_clusters,
            max_iter=max_iter,
            init="build",
        )

        # Map medoid indices back to full matrix
        full_medoids = sample_idx[sub_result.medoid_indices]

        # Evaluate on full dataset
        full_labels = _assign_clusters(dist_array, full_medoids)
        full_cost = _compute_cost(dist_array, full_medoids, full_labels)

        if full_cost < best_cost:
            best_cost = full_cost
            best_medoids = full_medoids.copy()
            best_sample_cost = sub_result.total_cost

    # Final assignment with best medoids
    assert best_medoids is not None
    final_labels = _assign_clusters(dist_array, best_medoids)

    return CLARAClusteringResult(
        labels=final_labels,
        medoid_indices=best_medoids,
        n_clusters=n_clusters,
        total_cost=best_cost,
        n_samples=n_samples,
        sample_size=sample_size,
        best_sample_cost=best_sample_cost,
    )
