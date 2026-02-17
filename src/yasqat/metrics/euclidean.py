"""Euclidean distance metric for sequence comparison.

Euclidean distance compares sequences based on the proportion of time spent
in each state, using L2 (Euclidean) distance on proportion vectors.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _compute_proportions(seq: np.ndarray, n_states: int) -> np.ndarray:
    """
    Compute state proportions for a sequence.

    Args:
        seq: Sequence (integer-encoded).
        n_states: Total number of possible states.

    Returns:
        Array of proportions for each state (sums to 1.0).
    """
    counts = np.zeros(n_states, dtype=np.float64)
    n = len(seq)
    for i in range(n):
        counts[seq[i]] += 1
    if n > 0:
        for i in range(n_states):
            counts[i] /= n
    return counts


@numba.jit(nopython=True, cache=True)
def _euclidean_kernel(props_a: np.ndarray, props_b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two proportion vectors.

    Args:
        props_a: Proportions for first sequence.
        props_b: Proportions for second sequence.

    Returns:
        Euclidean (L2) distance.
    """
    total = 0.0
    for i in range(len(props_a)):
        diff = props_a[i] - props_b[i]
        total += diff * diff
    return total**0.5


def euclidean_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    n_states: int | None = None,
    normalize: bool = False,
) -> float:
    """
    Compute Euclidean distance between two sequences based on state proportions.

    Both sequences are converted to proportion vectors (proportion of time
    spent in each state), then the L2 distance between these vectors is computed.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        n_states: Total number of possible states. If None, inferred
            from the maximum value in both sequences.
        normalize: If True, normalize by sqrt(2) (the maximum possible
            Euclidean distance between two proportion vectors).

    Returns:
        Euclidean distance (>= 0).

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 1, 2])
        >>> seq2 = np.array([0, 1, 1, 2, 2])
        >>> euclidean_distance(seq1, seq2)  # doctest: +SKIP
    """
    # Handle empty sequences
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    # Determine number of states
    if n_states is None:
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1

    # Compute proportions
    props_a = _compute_proportions(seq_a, n_states)
    props_b = _compute_proportions(seq_b, n_states)

    # Compute Euclidean distance
    distance = _euclidean_kernel(props_a, props_b)

    if normalize:
        # Max Euclidean distance between two probability vectors is sqrt(2)
        distance /= 2.0**0.5

    return float(distance)


class EuclideanMetric:
    """Euclidean distance metric class for sequence comparison."""

    name = "euclidean"

    def __init__(
        self,
        n_states: int | None = None,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the Euclidean metric.

        Args:
            n_states: Total number of possible states.
            normalize: Whether to normalize distances.
        """
        self.n_states = n_states
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute Euclidean distance between two sequences.

        Args:
            seq_a: First sequence (integer-encoded numpy array).
            seq_b: Second sequence (integer-encoded numpy array).

        Returns:
            Distance value (0 = identical sequences).
        """
        return euclidean_distance(
            seq_a,
            seq_b,
            n_states=self.n_states,
            normalize=self.normalize,
        )
