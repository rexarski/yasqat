"""PAM (Partitioning Around Medoids) clustering for sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    import polars as pl

    from yasqat.metrics.base import DistanceMatrix


@dataclass
class PAMClusteringResult:
    """Result of PAM clustering."""

    labels: np.ndarray
    """Cluster labels for each sequence."""

    medoid_indices: np.ndarray
    """Indices of medoid sequences."""

    n_clusters: int
    """Number of clusters."""

    sequence_ids: list[int | str]
    """Sequence IDs corresponding to each label."""

    total_cost: float
    """Total distance from all points to their medoids."""

    n_iterations: int
    """Number of iterations until convergence."""

    def cluster_sizes(self) -> dict[int, int]:
        """Return the size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts, strict=True))

    def get_cluster_members(self, cluster_id: int) -> list[int | str]:
        """Return sequence IDs in a specific cluster."""
        mask = self.labels == cluster_id
        return [self.sequence_ids[i] for i in range(len(mask)) if mask[i]]

    def get_medoid_ids(self) -> list[int | str]:
        """Return the sequence IDs of medoids."""
        return [self.sequence_ids[i] for i in self.medoid_indices]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to a polars DataFrame with id, cluster, and is_medoid columns."""
        import polars as pl

        is_medoid = np.zeros(len(self.labels), dtype=bool)
        is_medoid[self.medoid_indices] = True

        return pl.DataFrame(
            {
                "id": self.sequence_ids,
                "cluster": self.labels.tolist(),
                "is_medoid": is_medoid.tolist(),
            }
        )


@numba.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def _compute_cost(
    dist_matrix: np.ndarray,
    medoids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute total cost (sum of distances to medoids)."""
    n = len(labels)
    total = 0.0
    for i in range(n):
        total += dist_matrix[i, medoids[labels[i]]]
    return total


@numba.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def _assign_clusters(
    dist_matrix: np.ndarray,
    medoids: np.ndarray,
) -> np.ndarray:
    """Assign each point to the nearest medoid."""
    n = dist_matrix.shape[0]
    k = len(medoids)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        min_dist = np.inf
        min_idx = 0
        for j in range(k):
            d = dist_matrix[i, medoids[j]]
            if d < min_dist:
                min_dist = d
                min_idx = j
        labels[i] = min_idx

    return labels


@numba.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def _pam_swap_step(
    dist_matrix: np.ndarray,
    medoids: np.ndarray,
    labels: np.ndarray,
    current_cost: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Perform one swap step of PAM algorithm.

    Try swapping each medoid with each non-medoid and keep the best improvement.

    Returns:
        Tuple of (new_medoids, new_labels, new_cost, improved).
    """
    n = dist_matrix.shape[0]
    k = len(medoids)

    best_medoids = medoids.copy()
    best_labels = labels.copy()
    best_cost = current_cost
    improved = False

    # Create set of medoid indices for fast lookup
    medoid_set = set(medoids)

    for m_idx in range(k):
        for candidate in range(n):
            if candidate in medoid_set:
                continue

            # Try swapping
            new_medoids = medoids.copy()
            new_medoids[m_idx] = candidate

            # Reassign clusters
            new_labels = _assign_clusters(dist_matrix, new_medoids)

            # Compute new cost
            new_cost = _compute_cost(dist_matrix, new_medoids, new_labels)

            if new_cost < best_cost:
                best_medoids = new_medoids.copy()
                best_labels = new_labels.copy()
                best_cost = new_cost
                improved = True

    return best_medoids, best_labels, best_cost, improved


def _initialize_medoids(
    dist_matrix: np.ndarray,
    n_clusters: int,
    init: str = "build",
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialize medoids using specified method.

    Args:
        dist_matrix: Distance matrix.
        n_clusters: Number of clusters.
        init: Initialization method ("build", "random", or "k-medoids++").
        random_state: Random state for reproducibility.

    Returns:
        Array of initial medoid indices.
    """
    n = dist_matrix.shape[0]

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if init == "random":
        return rng.choice(n, n_clusters, replace=False).astype(np.int32)

    elif init == "build":
        # PAM BUILD: Greedy selection minimizing total distance
        medoids = []

        # First medoid: point with minimum total distance to all others
        total_dists = dist_matrix.sum(axis=1)
        first_medoid = int(np.argmin(total_dists))
        medoids.append(first_medoid)

        # Subsequent medoids: maximize reduction in total distance
        while len(medoids) < n_clusters:
            best_gain = -np.inf
            best_candidate = -1

            for candidate in range(n):
                if candidate in medoids:
                    continue

                # Compute gain from adding this candidate
                gain = 0.0
                for i in range(n):
                    if i in medoids or i == candidate:
                        continue
                    # Current distance to nearest medoid
                    current_dist = min(dist_matrix[i, m] for m in medoids)
                    # New distance if candidate added
                    new_dist = min(current_dist, dist_matrix[i, candidate])
                    gain += current_dist - new_dist

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            medoids.append(best_candidate)

        return np.array(medoids, dtype=np.int32)

    elif init == "k-medoids++":
        # Similar to k-means++: probabilistic selection
        medoids = []

        # First medoid: random
        first_medoid = int(rng.integers(0, n))
        medoids.append(first_medoid)

        # Subsequent medoids: probability proportional to squared distance
        while len(medoids) < n_clusters:
            # Compute distance to nearest medoid for each point
            min_dists = np.array(
                [min(dist_matrix[i, m] for m in medoids) for i in range(n)]
            )
            min_dists[medoids] = 0  # Exclude existing medoids

            # Square distances for probability weighting
            probs = min_dists**2
            if probs.sum() == 0:
                # All points are medoids or zero distance
                remaining = [i for i in range(n) if i not in medoids]
                if remaining:
                    next_medoid = rng.choice(remaining)
                else:
                    break
            else:
                probs /= probs.sum()
                next_medoid = rng.choice(n, p=probs)

            medoids.append(int(next_medoid))

        return np.array(medoids, dtype=np.int32)

    else:
        raise ValueError(f"Unknown initialization method: {init}")


def pam_clustering(
    distance_matrix: DistanceMatrix | np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    init: str = "build",
    random_state: int | np.random.Generator | None = None,
    sequence_ids: list[int | str] | None = None,
) -> PAMClusteringResult:
    """
    Perform PAM (Partitioning Around Medoids) clustering.

    PAM is a k-medoids algorithm that selects actual data points as cluster
    centers (medoids). Unlike k-means, it works directly with a distance
    matrix and is more robust to outliers.

    Args:
        distance_matrix: Pairwise distance matrix (DistanceMatrix or numpy array).
        n_clusters: Number of clusters to form.
        max_iter: Maximum number of iterations.
        init: Initialization method:
            - "build": PAM BUILD algorithm (greedy, deterministic).
            - "random": Random selection.
            - "k-medoids++": k-means++ style probabilistic selection.
        random_state: Random state for reproducibility (used if init != "build").
        sequence_ids: Optional list of sequence identifiers.

    Returns:
        PAMClusteringResult with cluster labels and medoid information.

    Example:
        >>> import numpy as np
        >>> from yasqat.clustering import pam_clustering
        >>> dist = np.array([
        ...     [0, 1, 4, 5],
        ...     [1, 0, 4, 5],
        ...     [4, 4, 0, 1],
        ...     [5, 5, 1, 0],
        ... ])
        >>> result = pam_clustering(dist, n_clusters=2)
        >>> result.labels
        array([0, 0, 1, 1])
    """
    # Extract numpy array from DistanceMatrix if needed
    if hasattr(distance_matrix, "matrix"):
        dist_array = distance_matrix.matrix
    else:
        dist_array = distance_matrix

    # Ensure matrix is square
    n = dist_array.shape[0]
    if dist_array.shape != (n, n):
        raise ValueError("Distance matrix must be square")

    if n_clusters > n:
        raise ValueError(
            f"Number of clusters ({n_clusters}) cannot exceed number of samples ({n})"
        )

    # Set default sequence IDs
    if sequence_ids is None:
        sequence_ids = list(range(n))
    elif len(sequence_ids) != n:
        raise ValueError(
            f"Number of sequence IDs ({len(sequence_ids)}) "
            f"must match distance matrix size ({n})"
        )

    # Ensure proper dtype
    dist_array = np.array(dist_array, dtype=np.float64)

    # Initialize medoids
    medoids = _initialize_medoids(dist_array, n_clusters, init, random_state)

    # Initial assignment
    labels = _assign_clusters(dist_array, medoids)
    current_cost = _compute_cost(dist_array, medoids, labels)

    # PAM SWAP iterations
    n_iter = 0
    for _n_iter in range(max_iter):
        medoids, labels, new_cost, improved = _pam_swap_step(
            dist_array, medoids, labels, current_cost
        )

        n_iter = _n_iter + 1
        if not improved:
            break

        current_cost = new_cost

    return PAMClusteringResult(
        labels=labels,
        medoid_indices=medoids,
        n_clusters=n_clusters,
        sequence_ids=sequence_ids,
        total_cost=current_cost,
        n_iterations=n_iter + 1,
    )


class PAMClustering:
    """PAM (Partitioning Around Medoids) clustering algorithm class."""

    name = "pam"

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        init: str = "build",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize PAM clustering.

        Args:
            n_clusters: Number of clusters to form.
            max_iter: Maximum number of iterations.
            init: Initialization method ("build", "random", "k-medoids++").
            random_state: Random state for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self._result: PAMClusteringResult | None = None

    def fit(
        self,
        distance_matrix: DistanceMatrix | np.ndarray,
        sequence_ids: list[int | str] | None = None,
    ) -> PAMClusteringResult:
        """
        Fit the clustering model to a distance matrix.

        Args:
            distance_matrix: Pairwise distance matrix.
            sequence_ids: Optional sequence identifiers.

        Returns:
            PAMClusteringResult with cluster assignments.
        """
        self._result = pam_clustering(
            distance_matrix,
            self.n_clusters,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
            sequence_ids=sequence_ids,
        )
        return self._result

    @property
    def labels(self) -> np.ndarray | None:
        """Return cluster labels from the last fit."""
        return self._result.labels if self._result else None

    @property
    def medoid_indices(self) -> np.ndarray | None:
        """Return medoid indices from the last fit."""
        return self._result.medoid_indices if self._result else None

    @property
    def result(self) -> PAMClusteringResult | None:
        """Return the full result from the last fit."""
        return self._result
