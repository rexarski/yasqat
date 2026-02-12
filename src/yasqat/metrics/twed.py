"""Time Warp Edit Distance (TWED) metric.

TWED combines edit distance with temporal elasticity, penalizing both
state mismatches and temporal gaps. See Marteau (2009).
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _twed_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    timestamps_a: np.ndarray,
    timestamps_b: np.ndarray,
    sub_costs: np.ndarray,
    nu: float,
    lmbda: float,
) -> float:
    """
    Numba-optimized TWED computation.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        timestamps_a: Timestamps for first sequence.
        timestamps_b: Timestamps for second sequence.
        sub_costs: Substitution cost matrix.
        nu: Stiffness parameter (>= 0). Controls elasticity penalty.
        lmbda: Deletion/insertion penalty (>= 0).

    Returns:
        TWED distance.
    """
    n = len(seq_a)
    m = len(seq_b)

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    # Initialize first column (delete from seq_a)
    for i in range(1, n + 1):
        if i == 1:
            dp[i, 0] = (
                sub_costs[seq_a[i - 1], seq_a[i - 1]]
                + nu * abs(timestamps_a[i - 1])
                + lmbda
            )
        else:
            dp[i, 0] = (
                dp[i - 1, 0]
                + sub_costs[seq_a[i - 1], seq_a[i - 2]]
                + nu * abs(timestamps_a[i - 1] - timestamps_a[i - 2])
                + lmbda
            )

    # Initialize first row (delete from seq_b)
    for j in range(1, m + 1):
        if j == 1:
            dp[0, j] = (
                sub_costs[seq_b[j - 1], seq_b[j - 1]]
                + nu * abs(timestamps_b[j - 1])
                + lmbda
            )
        else:
            dp[0, j] = (
                dp[0, j - 1]
                + sub_costs[seq_b[j - 1], seq_b[j - 2]]
                + nu * abs(timestamps_b[j - 1] - timestamps_b[j - 2])
                + lmbda
            )

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Match/substitute: align seq_a[i-1] with seq_b[j-1]
            cost_match = sub_costs[seq_a[i - 1], seq_b[j - 1]]
            if i > 1 and j > 1:
                cost_match += sub_costs[seq_a[i - 2], seq_b[j - 2]]
            time_diff_match = abs(timestamps_a[i - 1] - timestamps_b[j - 1])
            if i > 1 and j > 1:
                time_diff_match += abs(timestamps_a[i - 2] - timestamps_b[j - 2])
            d_match = dp[i - 1, j - 1] + cost_match + nu * time_diff_match

            # Delete from seq_a
            cost_del_a = sub_costs[seq_a[i - 1], seq_a[i - 2]] if i > 1 else 0.0
            if i > 1:
                time_diff_a = abs(timestamps_a[i - 1] - timestamps_a[i - 2])
            else:
                time_diff_a = abs(timestamps_a[i - 1])
            d_del_a = dp[i - 1, j] + cost_del_a + nu * time_diff_a + lmbda

            # Delete from seq_b
            cost_del_b = sub_costs[seq_b[j - 1], seq_b[j - 2]] if j > 1 else 0.0
            if j > 1:
                time_diff_b = abs(timestamps_b[j - 1] - timestamps_b[j - 2])
            else:
                time_diff_b = abs(timestamps_b[j - 1])
            d_del_b = dp[i, j - 1] + cost_del_b + nu * time_diff_b + lmbda

            dp[i, j] = min(d_match, d_del_a, d_del_b)

    return dp[n, m]


def twed_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    nu: float = 0.001,
    lmbda: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 1.0,
    timestamps_a: np.ndarray | None = None,
    timestamps_b: np.ndarray | None = None,
) -> float:
    """
    Compute Time Warp Edit Distance between two sequences.

    TWED combines edit distance with temporal elasticity. The stiffness
    parameter (nu) penalizes temporal distortion, while lambda penalizes
    deletions/insertions. See Marteau (2009).

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        nu: Stiffness parameter (>= 0). Higher values penalize temporal
            distortion more. Default: 0.001.
        lmbda: Deletion/insertion penalty (>= 0). Default: 1.0.
        sm: Substitution cost matrix or method ("constant").
        sub_cost: Cost for non-matching elements when sm="constant".
        timestamps_a: Timestamps for first sequence. If None, uses 0..n-1.
        timestamps_b: Timestamps for second sequence. If None, uses 0..m-1.

    Returns:
        TWED distance (0 = identical sequences at identical times).

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 2])
        >>> seq2 = np.array([0, 1, 1, 2])
        >>> twed_distance(seq1, seq2)  # doctest: +SKIP
        1.002
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    # Build substitution matrix if needed
    if isinstance(sm, str):
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        if sm == "constant":
            sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
            np.fill_diagonal(sm_matrix, 0.0)
        else:
            raise ValueError(f"Unknown substitution method: {sm}")
    else:
        sm_matrix = sm.astype(np.float64)

    # Default timestamps: 0, 1, 2, ...
    if timestamps_a is None:
        timestamps_a = np.arange(len(seq_a), dtype=np.float64)
    else:
        timestamps_a = timestamps_a.astype(np.float64)

    if timestamps_b is None:
        timestamps_b = np.arange(len(seq_b), dtype=np.float64)
    else:
        timestamps_b = timestamps_b.astype(np.float64)

    return float(
        _twed_kernel(seq_a, seq_b, timestamps_a, timestamps_b, sm_matrix, nu, lmbda)
    )


class TWEDMetric:
    """Time Warp Edit Distance metric class."""

    name = "twed"

    def __init__(
        self,
        nu: float = 0.001,
        lmbda: float = 1.0,
        sm: np.ndarray | str = "constant",
        sub_cost: float = 1.0,
    ) -> None:
        self.nu = nu
        self.lmbda = lmbda
        self.sm = sm
        self.sub_cost = sub_cost

    def compute(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        timestamps_a: np.ndarray | None = None,
        timestamps_b: np.ndarray | None = None,
    ) -> float:
        return twed_distance(
            seq_a,
            seq_b,
            nu=self.nu,
            lmbda=self.lmbda,
            sm=self.sm,
            sub_cost=self.sub_cost,
            timestamps_a=timestamps_a,
            timestamps_b=timestamps_b,
        )
