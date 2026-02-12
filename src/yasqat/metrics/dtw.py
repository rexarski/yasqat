"""Dynamic Time Warping (DTW) distance metric."""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _dtw_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sub_costs: np.ndarray,
    window: int,
) -> float:
    """
    Numba-optimized DTW computation with Sakoe-Chiba band.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        sub_costs: Substitution cost matrix.
        window: Sakoe-Chiba band width (0 = no constraint).

    Returns:
        DTW distance.
    """
    n = len(seq_a)
    m = len(seq_b)

    # Initialize with infinity
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    # Determine window constraint
    if window <= 0:
        w = max(n, m)
    else:
        w = window

    # Fill matrix with Sakoe-Chiba band
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = sub_costs[seq_a[i - 1], seq_b[j - 1]]
            dp[i, j] = cost + min(
                dp[i - 1, j],  # Insertion
                dp[i, j - 1],  # Deletion
                dp[i - 1, j - 1],  # Match
            )

    return dp[n, m]


@numba.jit(nopython=True, cache=True)
def _dtw_kernel_with_time(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    times_a: np.ndarray,
    times_b: np.ndarray,
    sub_costs: np.ndarray,
    window: int,
    max_time_diff: float,
) -> float:
    """
    Numba-optimized DTW with time difference constraints.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        times_a: Time points for first sequence.
        times_b: Time points for second sequence.
        sub_costs: Substitution cost matrix.
        window: Sakoe-Chiba band width (0 = no constraint).
        max_time_diff: Maximum allowed time difference (0 = no constraint).

    Returns:
        DTW distance.
    """
    n = len(seq_a)
    m = len(seq_b)

    # Initialize with infinity
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    # Determine window constraint
    if window <= 0:
        w = max(n, m)
    else:
        w = window

    # Fill matrix with Sakoe-Chiba band and time constraints
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            # Check time difference constraint
            if max_time_diff > 0:
                time_diff = abs(times_a[i - 1] - times_b[j - 1])
                if time_diff > max_time_diff:
                    continue

            cost = sub_costs[seq_a[i - 1], seq_b[j - 1]]
            dp[i, j] = cost + min(
                dp[i - 1, j],  # Insertion
                dp[i, j - 1],  # Deletion
                dp[i - 1, j - 1],  # Match
            )

    return dp[n, m]


@numba.jit(nopython=True, cache=True)
def _dtw_path_length(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sub_costs: np.ndarray,
    window: int,
) -> tuple[float, int]:
    """
    Compute DTW distance and warping path length.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        sub_costs: Substitution cost matrix.
        window: Sakoe-Chiba band width (0 = no constraint).

    Returns:
        Tuple of (DTW distance, path length).
    """
    n = len(seq_a)
    m = len(seq_b)

    # Initialize with infinity
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    path_len = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    # Determine window constraint
    if window <= 0:
        w = max(n, m)
    else:
        w = window

    # Fill matrices
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = sub_costs[seq_a[i - 1], seq_b[j - 1]]

            # Find minimum predecessor
            candidates = np.array(
                [
                    dp[i - 1, j],
                    dp[i, j - 1],
                    dp[i - 1, j - 1],
                ]
            )
            min_idx = np.argmin(candidates)

            if min_idx == 0:
                dp[i, j] = cost + dp[i - 1, j]
                path_len[i, j] = path_len[i - 1, j] + 1
            elif min_idx == 1:
                dp[i, j] = cost + dp[i, j - 1]
                path_len[i, j] = path_len[i, j - 1] + 1
            else:
                dp[i, j] = cost + dp[i - 1, j - 1]
                path_len[i, j] = path_len[i - 1, j - 1] + 1

    return dp[n, m], path_len[n, m]


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    window: int = 0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 1.0,
    normalize: bool = False,
    times_a: np.ndarray | None = None,
    times_b: np.ndarray | None = None,
    max_time_diff: float = 0.0,
) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    DTW finds the optimal alignment between two sequences that may vary
    in speed. It is particularly useful when sequences have different
    lengths or when the timing of events varies.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        window: Sakoe-Chiba band width for constraining warping.
            0 means no constraint (full DTW). Default: 0.
        sm: Substitution cost matrix or method.
            - numpy array: Use directly as substitution costs.
            - "constant": Use constant cost for non-matching elements.
            - "binary": Use 0 for match, 1 for non-match.
        sub_cost: Cost for non-matching elements when sm="constant".
        normalize: If True, normalize by warping path length.
        times_a: Optional time points for first sequence.
        times_b: Optional time points for second sequence.
        max_time_diff: Maximum allowed time difference between matched
            elements. 0 means no constraint. Requires times_a and times_b.

    Returns:
        DTW distance (0 = identical sequences with optimal warping).

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 2, 2])
        >>> seq2 = np.array([0, 1, 2])
        >>> dtw_distance(seq1, seq2)
        0.0
        >>> seq3 = np.array([0, 1, 3])
        >>> dtw_distance(seq1, seq3)
        1.0
    """
    # Handle empty sequences
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
        elif sm == "binary":
            sm_matrix = np.ones((n_states, n_states), dtype=np.float64)
            np.fill_diagonal(sm_matrix, 0.0)
        else:
            raise ValueError(f"Unknown substitution method: {sm}")
    else:
        sm_matrix = sm.astype(np.float64)

    # Choose kernel based on whether time constraints are used
    if times_a is not None and times_b is not None and max_time_diff > 0:
        times_a_arr = times_a.astype(np.float64)
        times_b_arr = times_b.astype(np.float64)
        distance = _dtw_kernel_with_time(
            seq_a, seq_b, times_a_arr, times_b_arr, sm_matrix, window, max_time_diff
        )
    elif normalize:
        distance, path_len = _dtw_path_length(seq_a, seq_b, sm_matrix, window)
        if path_len > 0:
            distance /= path_len
        return float(distance)
    else:
        distance = _dtw_kernel(seq_a, seq_b, sm_matrix, window)

    if normalize and not (
        times_a is not None and times_b is not None and max_time_diff > 0
    ):
        # Already handled above for standard normalization
        pass

    return float(distance)


class DTWMetric:
    """Dynamic Time Warping distance metric class."""

    name = "dtw"

    def __init__(
        self,
        window: int = 0,
        sm: np.ndarray | str = "constant",
        sub_cost: float = 1.0,
        normalize: bool = False,
        max_time_diff: float = 0.0,
    ) -> None:
        """
        Initialize the DTW metric.

        Args:
            window: Sakoe-Chiba band width (0 = no constraint).
            sm: Substitution cost matrix or method.
            sub_cost: Cost for non-matching elements.
            normalize: Whether to normalize by path length.
            max_time_diff: Maximum time difference constraint.
        """
        self.window = window
        self.sm = sm
        self.sub_cost = sub_cost
        self.normalize = normalize
        self.max_time_diff = max_time_diff

    def compute(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        times_a: np.ndarray | None = None,
        times_b: np.ndarray | None = None,
    ) -> float:
        """
        Compute DTW distance between two sequences.

        Args:
            seq_a: First sequence (integer-encoded).
            seq_b: Second sequence (integer-encoded).
            times_a: Optional time points for first sequence.
            times_b: Optional time points for second sequence.

        Returns:
            DTW distance.
        """
        return dtw_distance(
            seq_a,
            seq_b,
            window=self.window,
            sm=self.sm,
            sub_cost=self.sub_cost,
            normalize=self.normalize,
            times_a=times_a,
            times_b=times_b,
            max_time_diff=self.max_time_diff,
        )
