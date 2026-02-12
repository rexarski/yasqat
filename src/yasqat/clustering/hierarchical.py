"""Hierarchical clustering for sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import polars as pl

    from yasqat.metrics.base import DistanceMatrix


LinkageMethod = Literal["ward", "complete", "average", "single"]


@dataclass
class HierarchicalClusteringResult:
    """Result of hierarchical clustering."""

    labels: np.ndarray
    """Cluster labels for each sequence."""

    n_clusters: int
    """Number of clusters."""

    linkage_matrix: np.ndarray
    """Scipy linkage matrix for dendrogram visualization."""

    sequence_ids: list[int | str]
    """Sequence IDs corresponding to each label."""

    def cluster_sizes(self) -> dict[int, int]:
        """Return the size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts, strict=True))

    def get_cluster_members(self, cluster_id: int) -> list[int | str]:
        """Return sequence IDs in a specific cluster."""
        mask = self.labels == cluster_id
        return [self.sequence_ids[i] for i in range(len(mask)) if mask[i]]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to a polars DataFrame with id and cluster columns."""
        import polars as pl

        return pl.DataFrame(
            {
                "id": self.sequence_ids,
                "cluster": self.labels.tolist(),
            }
        )


def hierarchical_clustering(
    distance_matrix: DistanceMatrix | np.ndarray,
    n_clusters: int,
    method: LinkageMethod = "ward",
    sequence_ids: list[int | str] | None = None,
) -> HierarchicalClusteringResult:
    """
    Perform hierarchical agglomerative clustering on sequences.

    Uses scipy's hierarchical clustering with a precomputed distance matrix.

    Args:
        distance_matrix: Pairwise distance matrix (DistanceMatrix or numpy array).
        n_clusters: Number of clusters to form.
        method: Linkage method. One of:
            - "ward": Minimize within-cluster variance (requires Euclidean-like distances).
            - "complete": Maximum distance between cluster members.
            - "average": Average distance between cluster members.
            - "single": Minimum distance between cluster members.
        sequence_ids: Optional list of sequence identifiers. If not provided,
            uses 0-indexed integers.

    Returns:
        HierarchicalClusteringResult with cluster labels and metadata.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering import hierarchical_clustering
        >>> # Create a simple distance matrix
        >>> dist = np.array([
        ...     [0, 1, 4, 5],
        ...     [1, 0, 4, 5],
        ...     [4, 4, 0, 1],
        ...     [5, 5, 1, 0],
        ... ])
        >>> result = hierarchical_clustering(dist, n_clusters=2)
        >>> result.labels
        array([0, 0, 1, 1])
    """
    from scipy.cluster.hierarchy import (  # type: ignore[import-untyped]
        fcluster,
        linkage,
    )
    from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

    # Extract numpy array from DistanceMatrix if needed
    if hasattr(distance_matrix, "matrix"):
        dist_array = distance_matrix.matrix
    else:
        dist_array = distance_matrix

    # Ensure matrix is square and symmetric
    n = dist_array.shape[0]
    if dist_array.shape != (n, n):
        raise ValueError("Distance matrix must be square")

    # Set default sequence IDs
    if sequence_ids is None:
        sequence_ids = list(range(n))
    elif len(sequence_ids) != n:
        raise ValueError(
            f"Number of sequence IDs ({len(sequence_ids)}) "
            f"must match distance matrix size ({n})"
        )

    # Convert to condensed form for scipy
    # Make sure diagonal is 0 and matrix is symmetric
    dist_array = np.array(dist_array, dtype=np.float64)
    np.fill_diagonal(dist_array, 0)
    dist_array = (dist_array + dist_array.T) / 2  # Ensure symmetry

    condensed = squareform(dist_array, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed, method=method)

    # Cut tree to get cluster labels
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1

    return HierarchicalClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        linkage_matrix=linkage_matrix,
        sequence_ids=sequence_ids,
    )


class HierarchicalClustering:
    """Hierarchical clustering algorithm class."""

    name = "hierarchical"

    def __init__(
        self,
        n_clusters: int,
        method: LinkageMethod = "ward",
    ) -> None:
        """
        Initialize hierarchical clustering.

        Args:
            n_clusters: Number of clusters to form.
            method: Linkage method ("ward", "complete", "average", "single").
        """
        self.n_clusters = n_clusters
        self.method = method
        self._result: HierarchicalClusteringResult | None = None

    def fit(
        self,
        distance_matrix: DistanceMatrix | np.ndarray,
        sequence_ids: list[int | str] | None = None,
    ) -> HierarchicalClusteringResult:
        """
        Fit the clustering model to a distance matrix.

        Args:
            distance_matrix: Pairwise distance matrix.
            sequence_ids: Optional sequence identifiers.

        Returns:
            HierarchicalClusteringResult with cluster assignments.
        """
        self._result = hierarchical_clustering(
            distance_matrix,
            self.n_clusters,
            method=self.method,
            sequence_ids=sequence_ids,
        )
        return self._result

    @property
    def labels(self) -> np.ndarray | None:
        """Return cluster labels from the last fit."""
        return self._result.labels if self._result else None

    @property
    def result(self) -> HierarchicalClusteringResult | None:
        """Return the full result from the last fit."""
        return self._result
