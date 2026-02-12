"""Hamming distance metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _hamming_kernel(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Numba-optimized Hamming distance computation.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).

    Returns:
        Number of positions where the sequences differ.
    """
    n = len(seq_a)
    mismatches = 0

    for i in range(n):
        if seq_a[i] != seq_b[i]:
            mismatches += 1

    return mismatches


def hamming_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute Hamming distance between two sequences.

    The Hamming distance counts the number of positions where the
    corresponding elements are different. Both sequences must have
    the same length.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        normalize: If True, normalize by sequence length.

    Returns:
        Hamming distance (number of mismatches or normalized value).

    Raises:
        ValueError: If sequences have different lengths.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 2])
        >>> seq2 = np.array([0, 1, 1, 2])
        >>> hamming_distance(seq1, seq2)
        1.0
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(
            f"Sequences must have the same length for Hamming distance. "
            f"Got {len(seq_a)} and {len(seq_b)}"
        )

    distance = float(_hamming_kernel(seq_a, seq_b))

    if normalize and len(seq_a) > 0:
        distance /= len(seq_a)

    return distance


class HammingMetric:
    """Hamming distance metric class."""

    name = "hamming"

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the Hamming metric.

        Args:
            normalize: Whether to normalize distances by sequence length.
        """
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute distance between two sequences."""
        return hamming_distance(seq_a, seq_b, normalize=self.normalize)
