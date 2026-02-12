"""Base classes for sequence metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool


@dataclass
class DistanceMatrix:
    """
    Container for pairwise distance matrix.

    Attributes:
        values: Symmetric distance matrix as numpy array.
        labels: Optional labels for rows/columns (sequence IDs).
    """

    values: np.ndarray
    labels: list[int | str] | None = None

    def __post_init__(self) -> None:
        """Validate the distance matrix."""
        if self.values.ndim != 2:
            raise ValueError("Distance matrix must be 2-dimensional")
        if self.values.shape[0] != self.values.shape[1]:
            raise ValueError("Distance matrix must be square")

    def __getitem__(self, key: tuple[int, int]) -> float:
        """Get distance between two sequences by index."""
        return float(self.values[key])

    @property
    def n(self) -> int:
        """Number of sequences."""
        return int(self.values.shape[0])

    def get_distance(self, id1: int | str, id2: int | str) -> float:
        """Get distance between two sequences by label."""
        if self.labels is None:
            raise ValueError("Labels not set")
        i = self.labels.index(id1)
        j = self.labels.index(id2)
        return float(self.values[i, j])

    def to_condensed(self) -> np.ndarray:
        """Convert to condensed form (upper triangle, row-major)."""
        n = self.n
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(self.values[i, j])
        return np.array(condensed)

    @classmethod
    def from_condensed(
        cls,
        condensed: np.ndarray,
        labels: list[int | str] | None = None,
    ) -> DistanceMatrix:
        """Create from condensed form."""
        # Compute n from condensed length: n*(n-1)/2 = len
        n = int((1 + np.sqrt(1 + 8 * len(condensed))) / 2)
        values = np.zeros((n, n))

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                values[i, j] = condensed[idx]
                values[j, i] = condensed[idx]
                idx += 1

        return cls(values=values, labels=labels)


class SequenceMetric(ABC):
    """Abstract base class for sequence distance metrics."""

    name: str = "base"

    @abstractmethod
    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray, **kwargs: float) -> float:
        """
        Compute distance between two encoded sequences.

        Args:
            seq_a: First sequence as integer-encoded numpy array.
            seq_b: Second sequence as integer-encoded numpy array.
            **kwargs: Metric-specific parameters.

        Returns:
            Distance value (0 = identical).
        """

    def compute_matrix(self, pool: SequencePool, **kwargs: float) -> DistanceMatrix:
        """
        Compute pairwise distance matrix for a sequence pool.

        Args:
            pool: SequencePool containing sequences.
            **kwargs: Metric-specific parameters.

        Returns:
            DistanceMatrix with pairwise distances.
        """
        n = len(pool)
        ids = pool.sequence_ids
        values = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                seq_a = pool.get_encoded_sequence(ids[i])
                seq_b = pool.get_encoded_sequence(ids[j])
                dist = self.compute(seq_a, seq_b, **kwargs)
                values[i, j] = dist
                values[j, i] = dist

        return DistanceMatrix(values=values, labels=ids)


def build_substitution_matrix(
    n_states: int,
    method: str = "constant",
    cost: float = 2.0,
    transition_rates: np.ndarray | None = None,
    state_frequencies: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a substitution cost matrix.

    Args:
        n_states: Number of states in the alphabet.
        method: Method for computing costs.
            - "constant": All substitutions cost `cost`.
            - "trate": Costs based on transition rates (requires transition_rates).
            - "indels": Costs based on inverse state frequencies
              (c(a,b) = 1/freq(a) + 1/freq(b)). Requires state_frequencies.
            - "indelslog": Log variant (c(a,b) = log(1/freq(a)) + log(1/freq(b))).
              Requires state_frequencies.
            - "future": Chi-squared distance between next-state distributions.
              Requires transition_rates.
            - "features": Gower distance on user-defined state feature vectors.
              Requires state_frequencies as a (n_states, n_features) array.
        cost: Constant substitution cost (for "constant" method).
        transition_rates: Transition rate matrix (for "trate" and "future" methods).
        state_frequencies: Array of state frequencies/proportions (for "indels"
            and "indelslog" methods), or (n_states, n_features) feature matrix
            (for "features" method).

    Returns:
        Square matrix of substitution costs.
    """
    if method == "constant":
        result = np.full((n_states, n_states), cost)
        np.fill_diagonal(result, 0.0)
        return result

    if method == "trate":
        if transition_rates is None:
            raise ValueError("transition_rates required for 'trate' method")
        # Cost is inversely proportional to transition probability
        # c(a,b) = 2 - p(a->b) - p(b->a)
        result = 2.0 - transition_rates - transition_rates.T
        np.fill_diagonal(result, 0.0)
        return result

    if method == "indels":
        if state_frequencies is None:
            raise ValueError("state_frequencies required for 'indels' method")
        freq = np.asarray(state_frequencies, dtype=np.float64)
        # Avoid division by zero: replace zeros with a small value
        freq_safe = np.where(freq > 0, freq, 1e-10)
        inv_freq = 1.0 / freq_safe
        # c(a,b) = 1/freq(a) + 1/freq(b)
        result = inv_freq[:, None] + inv_freq[None, :]
        np.fill_diagonal(result, 0.0)
        return result

    if method == "indelslog":
        if state_frequencies is None:
            raise ValueError("state_frequencies required for 'indelslog' method")
        freq = np.asarray(state_frequencies, dtype=np.float64)
        freq_safe = np.where(freq > 0, freq, 1e-10)
        log_inv_freq = np.log(1.0 / freq_safe)
        # c(a,b) = log(1/freq(a)) + log(1/freq(b))
        result = log_inv_freq[:, None] + log_inv_freq[None, :]
        np.fill_diagonal(result, 0.0)
        return result

    if method == "future":
        if transition_rates is None:
            raise ValueError("transition_rates required for 'future' method")
        # Chi-squared distance between row distributions of transition matrix
        # c(a,b) = sum_k (p(a->k) - p(b->k))^2 / (p(a->k) + p(b->k))
        result = np.zeros((n_states, n_states), dtype=np.float64)
        for a in range(n_states):
            for b in range(a + 1, n_states):
                chi2_dist = 0.0
                for k in range(n_states):
                    total = transition_rates[a, k] + transition_rates[b, k]
                    if total > 0:
                        diff = transition_rates[a, k] - transition_rates[b, k]
                        chi2_dist += (diff * diff) / total
                result[a, b] = chi2_dist
                result[b, a] = chi2_dist
        return result

    if method == "features":
        if state_frequencies is None:
            raise ValueError(
                "state_frequencies required for 'features' method. "
                "Pass a (n_states, n_features) array of state feature vectors."
            )
        features = np.asarray(state_frequencies, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.shape[0] != n_states:
            raise ValueError(
                f"Feature matrix must have {n_states} rows, got {features.shape[0]}"
            )
        # Gower distance: mean absolute difference across features,
        # each normalized by feature range
        ranges = features.max(axis=0) - features.min(axis=0)
        ranges = np.where(ranges > 0, ranges, 1.0)  # avoid div by zero

        result = np.zeros((n_states, n_states), dtype=np.float64)
        for a in range(n_states):
            for b in range(a + 1, n_states):
                gower = float(np.mean(np.abs(features[a] - features[b]) / ranges))
                result[a, b] = gower
                result[b, a] = gower
        return result

    raise ValueError(f"Unknown method: {method}")
