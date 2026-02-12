"""Reverse Longest Common Prefix (RLCP) distance metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _rlcp_length_kernel(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Numba-optimized RLCP length computation.

    Iterates from the end of both sequences, counting matching suffix elements.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).

    Returns:
        Length of the longest common suffix.
    """
    n = len(seq_a)
    m = len(seq_b)
    min_len = min(n, m)

    rlcp_len = 0
    for i in range(1, min_len + 1):
        if seq_a[n - i] == seq_b[m - i]:
            rlcp_len += 1
        else:
            break

    return rlcp_len


def rlcp_length(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    Compute the length of the reverse longest common prefix (longest common suffix).

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Length of the longest common suffix.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([4, 5, 2, 3])
        >>> rlcp_length(seq1, seq2)
        2
    """
    return int(_rlcp_length_kernel(seq_a, seq_b))


def rlcp_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute RLCP-based distance between two sequences.

    The distance is computed as the sum of sequence lengths minus
    twice the RLCP length: d = len(a) + len(b) - 2 * RLCP(a, b)

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        normalize: If True, normalize by the sum of sequence lengths.

    Returns:
        RLCP-based distance.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([4, 5, 2, 3])
        >>> rlcp_distance(seq1, seq2)
        4.0
    """
    rlcp_len = rlcp_length(seq_a, seq_b)
    distance = float(len(seq_a) + len(seq_b) - 2 * rlcp_len)

    if normalize:
        total_len = len(seq_a) + len(seq_b)
        if total_len > 0:
            distance /= total_len

    return distance


def rlcp_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute RLCP-based similarity between two sequences.

    The similarity is the RLCP length normalized by the length of
    the shorter sequence: sim = RLCP(a, b) / min(len(a), len(b))

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        Similarity value between 0 and 1.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2, 3])
        >>> seq2 = np.array([4, 5, 2, 3])
        >>> rlcp_similarity(seq1, seq2)
        0.5
    """
    rlcp_len = rlcp_length(seq_a, seq_b)
    min_len = min(len(seq_a), len(seq_b))

    if min_len == 0:
        return 1.0 if len(seq_a) == 0 and len(seq_b) == 0 else 0.0

    return rlcp_len / min_len


class RLCPMetric:
    """Reverse Longest Common Prefix distance metric class."""

    name = "rlcp"

    def __init__(self, normalize: bool = False) -> None:
        """
        Initialize the RLCP metric.

        Args:
            normalize: Whether to normalize distances.
        """
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute distance between two sequences."""
        return rlcp_distance(seq_a, seq_b, normalize=self.normalize)
