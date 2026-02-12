"""Chi-squared (Chi2) distance metric for sequence comparison.

Chi2 distance compares sequences based on the distribution of time spent
in each state/category, rather than the order of states. This is useful
when the overall proportion of states matters more than their sequence.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _compute_state_counts(seq: np.ndarray, n_states: int) -> np.ndarray:
    """
    Count occurrences of each state in a sequence.

    Args:
        seq: Sequence (integer-encoded).
        n_states: Total number of possible states.

    Returns:
        Array of counts for each state.
    """
    counts = np.zeros(n_states, dtype=np.float64)
    for i in range(len(seq)):
        counts[seq[i]] += 1
    return counts


@numba.jit(nopython=True, cache=True)
def _chi2_distance_kernel(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """
    Compute Chi2 distance between two count vectors.

    Chi2 distance = 0.5 * sum((a_i - b_i)^2 / (a_i + b_i))

    Args:
        counts_a: State counts for first sequence.
        counts_b: State counts for second sequence.

    Returns:
        Chi2 distance.
    """
    distance = 0.0
    for i in range(len(counts_a)):
        total = counts_a[i] + counts_b[i]
        if total > 0:
            diff = counts_a[i] - counts_b[i]
            distance += (diff * diff) / total
    return 0.5 * distance


def chi2_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    n_states: int | None = None,
    normalize: bool = False,
) -> float:
    """
    Compute Chi-squared distance between two sequences.

    Chi2 distance compares the state distributions (proportion of time
    spent in each state) rather than the order of states. Two sequences
    with the same state proportions will have distance 0, regardless of
    the actual ordering.

    The Chi2 distance is defined as:
        d = 0.5 * sum((count_a[i] - count_b[i])^2 / (count_a[i] + count_b[i]))

    where count_a[i] and count_b[i] are the number of occurrences of
    state i in sequences a and b respectively.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        n_states: Total number of possible states. If None, inferred
            from the maximum value in both sequences.
        normalize: If True, normalize by the number of states.

    Returns:
        Chi2 distance (>= 0). Lower values indicate more similar distributions.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 1, 2])  # 2 zeros, 2 ones, 1 two
        >>> seq2 = np.array([0, 1, 1, 2, 2])  # 1 zero, 2 ones, 2 twos
        >>> chi2_distance(seq1, seq2)
        0.5
    """
    # Handle empty sequences
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        # If one is empty and other is not, return maximum distance
        # based on the non-empty sequence's variance
        return float("inf")

    # Determine number of states
    if n_states is None:
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1

    # Compute state counts
    counts_a = _compute_state_counts(seq_a, n_states)
    counts_b = _compute_state_counts(seq_b, n_states)

    # Compute Chi2 distance
    distance = _chi2_distance_kernel(counts_a, counts_b)

    if normalize:
        # Normalize by number of states with non-zero counts
        n_active = np.sum((counts_a + counts_b) > 0)
        if n_active > 0:
            distance /= n_active

    return float(distance)


def chi2_distance_weighted(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    weights: np.ndarray | None = None,
    n_states: int | None = None,
) -> float:
    """
    Compute weighted Chi-squared distance between two sequences.

    This variant allows different weights for different states, which
    can be useful when some states are more important than others.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        weights: Weight for each state. If None, uniform weights are used.
        n_states: Total number of possible states.

    Returns:
        Weighted Chi2 distance.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1])
        >>> seq2 = np.array([0, 1, 1])
        >>> weights = np.array([2.0, 1.0])  # State 0 is twice as important
        >>> chi2_distance_weighted(seq1, seq2, weights)
    """
    # Handle empty sequences
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    # Determine number of states
    if n_states is None:
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1

    # Default weights
    if weights is None:
        weights = np.ones(n_states, dtype=np.float64)
    elif len(weights) < n_states:
        # Extend weights with ones
        new_weights = np.ones(n_states, dtype=np.float64)
        new_weights[: len(weights)] = weights
        weights = new_weights

    # Compute state counts
    counts_a = _compute_state_counts(seq_a, n_states)
    counts_b = _compute_state_counts(seq_b, n_states)

    # Compute weighted Chi2 distance
    distance = 0.0
    for i in range(n_states):
        total = counts_a[i] + counts_b[i]
        if total > 0:
            diff = counts_a[i] - counts_b[i]
            distance += weights[i] * (diff * diff) / total

    return 0.5 * float(distance)


def state_distribution(seq: np.ndarray, n_states: int | None = None) -> np.ndarray:
    """
    Compute the state distribution (proportions) for a sequence.

    Args:
        seq: Sequence (integer-encoded numpy array).
        n_states: Total number of possible states.

    Returns:
        Array of proportions (sums to 1.0).

    Example:
        >>> import numpy as np
        >>> seq = np.array([0, 0, 1, 1, 1])
        >>> state_distribution(seq)
        array([0.4, 0.6])
    """
    if len(seq) == 0:
        if n_states is None:
            return np.array([], dtype=np.float64)
        return np.zeros(n_states, dtype=np.float64)

    if n_states is None:
        n_states = int(seq.max()) + 1

    counts = _compute_state_counts(seq, n_states)
    return counts / len(seq)


class Chi2Metric:
    """Chi-squared distance metric class."""

    name = "chi2"

    def __init__(
        self,
        n_states: int | None = None,
        normalize: bool = False,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the Chi2 metric.

        Args:
            n_states: Total number of possible states.
            normalize: Whether to normalize distances.
            weights: Optional state weights.
        """
        self.n_states = n_states
        self.normalize = normalize
        self.weights = weights

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute Chi2 distance between two sequences."""
        if self.weights is not None:
            return chi2_distance_weighted(
                seq_a,
                seq_b,
                weights=self.weights,
                n_states=self.n_states,
            )
        return chi2_distance(
            seq_a,
            seq_b,
            n_states=self.n_states,
            normalize=self.normalize,
        )
