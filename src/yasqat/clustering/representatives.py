"""Representative sequence extraction from clustering results.

Provides methods to select representative sequences from a distance matrix
using frequency, centrality, or density-based strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepresentativeResult:
    """Result of representative sequence extraction."""

    indices: np.ndarray
    """Indices of representative sequences in the distance matrix."""

    scores: np.ndarray
    """Score for each representative (interpretation depends on strategy)."""

    strategy: str
    """Strategy used for selection."""

    def __repr__(self) -> str:
        return (
            f"RepresentativeResult(n={len(self.indices)}, strategy='{self.strategy}')"
        )


def extract_representatives(
    dist_matrix: np.ndarray,
    n_representatives: int = 4,
    strategy: str = "centrality",
    labels: np.ndarray | None = None,
) -> RepresentativeResult:
    """
    Select representative sequences from a distance matrix.

    Three strategies are available:

    - **centrality**: Sequences with minimum total distance to all others
      (or to cluster members if labels provided). These are the most
      "central" or "typical" sequences.

    - **frequency**: Sequences appearing in the densest regions, measured
      by counting neighbors within a distance threshold.

    - **density**: Sequences in the highest-density regions, using a
      kernel density estimate based on the distance matrix.

    Args:
        dist_matrix: Symmetric pairwise distance matrix (n x n).
        n_representatives: Number of representatives to select.
        strategy: Selection strategy ("centrality", "frequency", "density").
        labels: Optional cluster labels. If provided, selects representatives
            per cluster (distributing n_representatives across clusters).

    Returns:
        RepresentativeResult with indices and scores of selected sequences.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering.representatives import extract_representatives
        >>> dist = np.array([
        ...     [0, 1, 4, 5],
        ...     [1, 0, 4, 5],
        ...     [4, 4, 0, 1],
        ...     [5, 5, 1, 0],
        ... ], dtype=float)
        >>> result = extract_representatives(dist, n_representatives=2)
        >>> len(result.indices)
        2
    """
    n = dist_matrix.shape[0]
    if n == 0:
        return RepresentativeResult(
            indices=np.array([], dtype=np.int64),
            scores=np.array([], dtype=np.float64),
            strategy=strategy,
        )

    n_representatives = min(n_representatives, n)

    if labels is not None:
        return _extract_per_cluster(dist_matrix, n_representatives, strategy, labels)

    if strategy == "centrality":
        return _centrality_representatives(dist_matrix, n_representatives)
    elif strategy == "frequency":
        return _frequency_representatives(dist_matrix, n_representatives)
    elif strategy == "density":
        return _density_representatives(dist_matrix, n_representatives)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            "Available: 'centrality', 'frequency', 'density'"
        )


def _centrality_representatives(
    dist_matrix: np.ndarray,
    n_representatives: int,
) -> RepresentativeResult:
    """Select representatives by minimum total distance (most central)."""
    total_dists = dist_matrix.sum(axis=1)
    indices = np.argsort(total_dists)[:n_representatives]
    scores = total_dists[indices]

    return RepresentativeResult(
        indices=indices,
        scores=scores,
        strategy="centrality",
    )


def _frequency_representatives(
    dist_matrix: np.ndarray,
    n_representatives: int,
) -> RepresentativeResult:
    """Select representatives from densest regions (most neighbors nearby)."""
    n = dist_matrix.shape[0]

    # Use median distance as threshold
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    if len(upper_tri) == 0:
        return RepresentativeResult(
            indices=np.arange(min(n_representatives, n)),
            scores=np.zeros(min(n_representatives, n)),
            strategy="frequency",
        )

    threshold = float(np.median(upper_tri))

    # Count neighbors within threshold for each point
    neighbor_counts = np.sum((dist_matrix <= threshold) & (dist_matrix > 0), axis=1)

    indices = np.argsort(-neighbor_counts)[:n_representatives]
    scores = neighbor_counts[indices].astype(np.float64)

    return RepresentativeResult(
        indices=indices,
        scores=scores,
        strategy="frequency",
    )


def _density_representatives(
    dist_matrix: np.ndarray,
    n_representatives: int,
) -> RepresentativeResult:
    """Select representatives using kernel density estimation on distances."""
    n = dist_matrix.shape[0]

    # Bandwidth: median of all pairwise distances
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    if len(upper_tri) == 0:
        return RepresentativeResult(
            indices=np.arange(min(n_representatives, n)),
            scores=np.zeros(min(n_representatives, n)),
            strategy="density",
        )

    bandwidth = float(np.median(upper_tri))
    if bandwidth == 0:
        bandwidth = 1.0

    # Gaussian kernel density: sum of K(d_ij / h) for each point i
    densities = np.sum(np.exp(-0.5 * (dist_matrix / bandwidth) ** 2), axis=1)

    indices = np.argsort(-densities)[:n_representatives]
    scores = densities[indices]

    return RepresentativeResult(
        indices=indices,
        scores=scores,
        strategy="density",
    )


def _extract_per_cluster(
    dist_matrix: np.ndarray,
    n_representatives: int,
    strategy: str,
    labels: np.ndarray,
) -> RepresentativeResult:
    """Extract representatives per cluster, distributing budget across clusters."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Distribute representatives across clusters
    base_per_cluster = n_representatives // n_clusters
    remainder = n_representatives % n_clusters

    all_indices = []
    all_scores = []

    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        n_k = base_per_cluster + (1 if i < remainder else 0)
        n_k = min(n_k, len(cluster_indices))

        if n_k == 0:
            continue

        # Extract sub-distance-matrix for this cluster
        cluster_dist = dist_matrix[np.ix_(cluster_indices, cluster_indices)]

        if strategy == "centrality":
            result = _centrality_representatives(cluster_dist, n_k)
        elif strategy == "frequency":
            result = _frequency_representatives(cluster_dist, n_k)
        elif strategy == "density":
            result = _density_representatives(cluster_dist, n_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Map back to original indices
        all_indices.extend(cluster_indices[result.indices].tolist())
        all_scores.extend(result.scores.tolist())

    return RepresentativeResult(
        indices=np.array(all_indices, dtype=np.int64),
        scores=np.array(all_scores, dtype=np.float64),
        strategy=strategy,
    )
