"""Optimal Matching (Edit Distance) metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _optimal_matching_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
) -> float:
    """
    Numba-optimized optimal matching computation.

    Uses dynamic programming to compute the edit distance between
    two sequences with custom substitution and indel costs.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        indel_cost: Cost of insertion/deletion.
        sub_costs: Substitution cost matrix.

    Returns:
        Optimal matching distance.
    """
    n = len(seq_a)
    m = len(seq_b)

    # DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    # Initialize borders
    for i in range(n + 1):
        dp[i, 0] = i * indel_cost
    for j in range(m + 1):
        dp[0, j] = j * indel_cost

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Get substitution cost from matrix
            sub_cost = sub_costs[seq_a[i - 1], seq_b[j - 1]]

            dp[i, j] = min(
                dp[i - 1, j] + indel_cost,  # Deletion
                dp[i, j - 1] + indel_cost,  # Insertion
                dp[i - 1, j - 1] + sub_cost,  # Substitution
            )

    return dp[n, m]


def optimal_matching_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    normalize: bool = False,
) -> float:
    """
    Compute optimal matching distance between two sequences.

    Optimal matching (OM) is an edit distance that measures the minimum
    cost of transforming one sequence into another using insertions,
    deletions, and substitutions.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost. Default: 1.0.
        sm: Substitution cost matrix or method.
            - numpy array: Use directly as substitution costs.
            - "constant": Use constant cost (specified by sub_cost).
        sub_cost: Constant substitution cost (used when sm="constant").
        normalize: If True, normalize by the maximum possible distance.

    Returns:
        Optimal matching distance (0 = identical sequences).

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 2])
        >>> seq2 = np.array([0, 1, 1, 2])
        >>> optimal_matching_distance(seq1, seq2)
        2.0
    """
    # Handle empty sequences
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0:
        return float(len(seq_b) * indel)
    if len(seq_b) == 0:
        return float(len(seq_a) * indel)

    # Build substitution matrix if needed
    if isinstance(sm, str):
        if sm == "constant":
            n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
            sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
            np.fill_diagonal(sm_matrix, 0.0)
        else:
            raise ValueError(f"Unknown substitution method: {sm}")
    else:
        sm_matrix = sm.astype(np.float64)

    distance = _optimal_matching_kernel(seq_a, seq_b, indel, sm_matrix)

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)


class OptimalMatchingMetric:
    """Optimal Matching distance metric class."""

    name = "optimal_matching"

    def __init__(
        self,
        indel: float = 1.0,
        sm: np.ndarray | str = "constant",
        sub_cost: float = 2.0,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the Optimal Matching metric.

        Args:
            indel: Insertion/deletion cost.
            sm: Substitution cost matrix or method.
            sub_cost: Constant substitution cost.
            normalize: Whether to normalize distances.
        """
        self.indel = indel
        self.sm = sm
        self.sub_cost = sub_cost
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute optimal matching distance between two sequences.

        Args:
            seq_a: First sequence (integer-encoded numpy array).
            seq_b: Second sequence (integer-encoded numpy array).

        Returns:
            Distance value (0 = identical sequences).
        """
        return optimal_matching_distance(
            seq_a,
            seq_b,
            indel=self.indel,
            sm=self.sm,
            sub_cost=self.sub_cost,
            normalize=self.normalize,
        )
