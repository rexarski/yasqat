"""Longest Common Prefix (LCP) distance metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _lcp_length_kernel(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Numba-optimized LCP length computation.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).

    Returns:
        Length of the longest common prefix.
    """
    n = len(seq_a)
    m = len(seq_b)
    min_len = min(n, m)

    lcp_len = 0
    for i in range(min_len):
        if seq_a[i] == seq_b[i]:
            lcp_len += 1
        else:
            break

    return lcp_len


def lcp_length(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Compute the length of the longest common prefix.

    The longest common prefix is the longest sequence of identical
    elements at the beginning of both sequences.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Length of the longest common prefix.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 1, 4, 5])
        >>> lcp_length(seq1, seq2)
        2
    """
    return int(_lcp_length_kernel(seq_a, seq_b))


def lcp_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute LCP-based distance between two sequences.

    The distance is computed as the sum of sequence lengths minus
    twice the LCP length: d = len(a) + len(b) - 2 * LCP(a, b)

    This measures how different the sequences are from their common prefix.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        normalize: If True, normalize by the sum of sequence lengths.

    Returns:
        LCP-based distance.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 1, 4, 5])
        >>> lcp_distance(seq1, seq2)
        4.0
    """
    lcp_len = lcp_length(seq_a, seq_b)
    distance = float(len(seq_a) + len(seq_b) - 2 * lcp_len)

    if normalize:
        total_len = len(seq_a) + len(seq_b)
        if total_len > 0:
            distance /= total_len

    return distance


def lcp_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute LCP-based similarity between two sequences.

    The similarity is the LCP length normalized by the length of
    the shorter sequence: sim = LCP(a, b) / min(len(a), len(b))

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Similarity value between 0 and 1.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([0, 1, 4, 5])
        >>> lcp_similarity(seq1, seq2)
        0.5
    """
    lcp_len = lcp_length(seq_a, seq_b)
    min_len = min(len(seq_a), len(seq_b))

    if min_len == 0:
        return 1.0 if len(seq_a) == 0 and len(seq_b) == 0 else 0.0

    return lcp_len / min_len


class LCPMetric:
    """Longest Common Prefix distance metric class."""

    name = "lcp"

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the LCP metric.

        Args:
            normalize: Whether to normalize distances.
        """
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute distance between two sequences."""
        return lcp_distance(seq_a, seq_b, normalize=self.normalize)
