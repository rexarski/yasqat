"""Longest Common Subsequence (LCS) distance metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _lcs_length_kernel(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Numba-optimized LCS length computation.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).

    Returns:
        Length of the longest common subsequence.
    """
    n = len(seq_a)
    m = len(seq_b)

    # DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    return dp[n, m]


def lcs_length(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Compute the length of the longest common subsequence.

    A subsequence is a sequence that can be derived from another
    sequence by deleting some or no elements without changing
    the order of the remaining elements.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Length of the longest common subsequence.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 2, 1, 3])
        >>> lcs_length(seq1, seq2)
        3
    """
    return int(_lcs_length_kernel(seq_a, seq_b))


def lcs_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute LCS-based distance between two sequences.

    The distance is computed as the sum of sequence lengths minus
    twice the LCS length: d = len(a) + len(b) - 2 * LCS(a, b)

    This represents the minimum number of deletions required to
    transform both sequences into their LCS.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        normalize: If True, normalize by the sum of sequence lengths.

    Returns:
        LCS-based distance.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 2, 1, 3])
        >>> lcs_distance(seq1, seq2)
        2.0
    """
    lcs_len = lcs_length(seq_a, seq_b)
    distance = float(len(seq_a) + len(seq_b) - 2 * lcs_len)

    if normalize:
        total_len = len(seq_a) + len(seq_b)
        if total_len > 0:
            distance /= total_len

    return distance


def lcs_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute LCS-based similarity between two sequences.

    The similarity is the LCS length normalized by the length of
    the longer sequence: sim = LCS(a, b) / max(len(a), len(b))

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Similarity value between 0 and 1.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 2, 1, 3])
        >>> lcs_similarity(seq1, seq2)
        0.75
    """
    lcs_len = lcs_length(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b))

    if max_len == 0:
        return 1.0

    return lcs_len / max_len


class LCSMetric:
    """Longest Common Subsequence distance metric class."""

    name = "lcs"

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the LCS metric.

        Args:
            normalize: Whether to normalize distances.
        """
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute distance between two sequences."""
        return lcs_distance(seq_a, seq_b, normalize=self.normalize)
