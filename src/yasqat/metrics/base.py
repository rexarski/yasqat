"""Distance matrix container and substitution-cost builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    @classmethod
    def coerce(cls, matrix: DistanceMatrix | np.ndarray) -> DistanceMatrix:
        """Normalize a raw numpy array or DistanceMatrix to a DistanceMatrix.

        The single seam through which distance consumers (clustering,
        discrepancy analysis, dissimilarity trees) accept either form:
        identity if already a ``DistanceMatrix``, otherwise the array is
        cast to float64 and validated (2-D, square) by construction.

        Args:
            matrix: A ``DistanceMatrix`` or square numpy array.

        Returns:
            A DistanceMatrix (labels are None when built from a raw array).
        """
        if isinstance(matrix, cls):
            return matrix
        return cls(values=np.asarray(matrix, dtype=np.float64))

    def __getitem__(self, key: tuple[int, int]) -> float:
        """Get distance between two sequences by index."""
        return float(self.values[key])

    @property
    def n(self) -> int:
        """Number of sequences."""
        return int(self.values.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the distance matrix."""
        return (self.values.shape[0], self.values.shape[1])

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
        method: Method for computing costs (TraMineR ``seqcost`` names,
            lower-cased). ``"constant"``: all substitutions cost ``cost``.
            ``"trate"``: ``2 - p(a,b) - p(b,a)`` from ``transition_rates``.
            ``"indels"``: per-state indel ``1 / freq(a)`` (states absent
            from the data get 1), ``c(a,b) = indel(a) + indel(b)``; needs
            ``state_frequencies``. ``"indelslog"``: per-state indel
            ``log(2 / (1 + freq(a)))``, summed likewise. ``"future"``:
            chi-square distance between the rows of the transition matrix,
            ``sqrt(sum_k (p(a,k) - p(b,k))^2 / colsum_k)``; needs
            ``transition_rates``. ``"features"``: Gower distance on
            user-defined state feature vectors passed as
            ``state_frequencies`` with shape ``(n_states, n_features)``.
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
        # seqcost.R: indels[is.na(indels)] <- 1 for states absent from the data.
        inv_freq = np.where(freq > 0, 1.0 / np.where(freq > 0, freq, 1.0), 1.0)
        # c(a,b) = 1/freq(a) + 1/freq(b)
        result = inv_freq[:, None] + inv_freq[None, :]
        np.fill_diagonal(result, 0.0)
        return result

    if method == "indelslog":
        if state_frequencies is None:
            raise ValueError("state_frequencies required for 'indelslog' method")
        freq = np.asarray(state_frequencies, dtype=np.float64)
        # seqcost.R: indels <- log(2 / (1 + freq)); absent states count as freq 1.
        freq_safe = np.where(freq > 0, freq, 1.0)
        log_indel = np.log(2.0 / (1.0 + freq_safe))
        result = log_indel[:, None] + log_indel[None, :]
        np.fill_diagonal(result, 0.0)
        return result

    if method == "future":
        if transition_rates is None:
            raise ValueError("transition_rates required for 'future' method")
        # seqcost.R chisqdista(): sqrt(sum_k (1 / colsum_k) * (p(a,k) - p(b,k))^2),
        # with a zero column sum contributing nothing (1 / Inf).
        rates = np.asarray(transition_rates, dtype=np.float64)
        col_sums = rates.sum(axis=0)
        weights = np.where(
            col_sums > 0, 1.0 / np.where(col_sums > 0, col_sums, 1.0), 0.0
        )
        result = np.zeros((n_states, n_states), dtype=np.float64)
        for a in range(n_states):
            for b in range(a + 1, n_states):
                diff = rates[a] - rates[b]
                chi2_dist = float(np.sqrt(np.sum(weights * diff * diff)))
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
