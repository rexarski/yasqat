"""Soft Dynamic Time Warping (SoftDTW) metric.

SoftDTW is a differentiable variant of DTW that can be used for gradient-based
optimization and machine learning applications.

Reference:
    Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss Function for
    Time-Series. In International Conference on Machine Learning (ICML).
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _softmin(values: np.ndarray, gamma: float) -> float:
    """
    Compute soft minimum using log-sum-exp trick.

    softmin(x; gamma) = -gamma * log(sum(exp(-x / gamma)))

    For numerical stability, we use:
    softmin(x) = min(x) - gamma * log(sum(exp(-(x - min(x)) / gamma)))

    Args:
        values: Array of values.
        gamma: Smoothing parameter (> 0). Smaller = harder minimum.

    Returns:
        Soft minimum value.
    """
    min_val = np.min(values)
    shifted = (values - min_val) / gamma
    log_sum_exp = np.log(np.sum(np.exp(-shifted)))
    return min_val - gamma * log_sum_exp


@numba.jit(nopython=True, cache=True)
def _softdtw_kernel(
    dist_matrix: np.ndarray,
    gamma: float,
) -> float:
    """
    Numba-optimized SoftDTW computation.

    Args:
        dist_matrix: Pairwise distance matrix between sequence elements.
        gamma: Smoothing parameter.

    Returns:
        SoftDTW distance.
    """
    n, m = dist_matrix.shape

    # DP table for soft DTW
    R = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i - 1, j - 1]

            # Soft minimum of three predecessors
            r0 = R[i - 1, j - 1]
            r1 = R[i - 1, j]
            r2 = R[i, j - 1]

            # Compute soft minimum
            predecessors = np.array([r0, r1, r2])
            soft_min = _softmin(predecessors, gamma)

            R[i, j] = cost + soft_min

    return float(R[n, m])


@numba.jit(nopython=True, cache=True)
def _softdtw_kernel_with_window(
    dist_matrix: np.ndarray,
    gamma: float,
    window: int,
) -> float:
    """
    Numba-optimized SoftDTW computation with Sakoe-Chiba band.

    Args:
        dist_matrix: Pairwise distance matrix between sequence elements.
        gamma: Smoothing parameter.
        window: Sakoe-Chiba band width.

    Returns:
        SoftDTW distance.
    """
    n, m = dist_matrix.shape

    # DP table
    R = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = dist_matrix[i - 1, j - 1]

            # Get predecessors (may be inf if outside band)
            r0 = R[i - 1, j - 1]
            r1 = R[i - 1, j]
            r2 = R[i, j - 1]

            # Compute soft minimum
            predecessors = np.array([r0, r1, r2])
            soft_min = _softmin(predecessors, gamma)

            R[i, j] = cost + soft_min

    return float(R[n, m])


@numba.jit(nopython=True, cache=True)
def _compute_pairwise_distances(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sub_costs: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise distance matrix between sequence elements.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        sub_costs: Substitution cost matrix.

    Returns:
        Distance matrix of shape (len(seq_a), len(seq_b)).
    """
    n = len(seq_a)
    m = len(seq_b)
    D = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            D[i, j] = sub_costs[seq_a[i], seq_b[j]]

    return D


def softdtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    gamma: float = 1.0,
    window: int = 0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 1.0,
    normalize: bool = False,
) -> float:
    """
    Compute Soft-DTW distance between two sequences.

    SoftDTW is a differentiable variant of DTW that replaces the hard minimum
    operation with a soft minimum (log-sum-exp). This makes it suitable for
    gradient-based optimization.

    As gamma approaches 0, SoftDTW converges to standard DTW.
    Larger gamma values make the distance "softer" and smoother.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        gamma: Smoothing parameter (> 0). Default: 1.0.
            - Small gamma (e.g., 0.01): Nearly equivalent to hard DTW.
            - Large gamma (e.g., 10.0): Very smooth, considers many alignments.
        window: Sakoe-Chiba band width (0 = no constraint). Default: 0.
        sm: Substitution cost matrix or method.
            - numpy array: Use directly as substitution costs.
            - "constant": Use constant cost for non-matching elements.
            - "squared": Use squared difference (for numeric sequences).
        sub_cost: Cost for non-matching elements when sm="constant".
        normalize: If True, normalize by sequence lengths.

    Returns:
        SoftDTW distance.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 0, 1, 2, 2])
        >>> seq2 = np.array([0, 1, 2])
        >>> softdtw_distance(seq1, seq2, gamma=1.0)  # Returns a float
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")

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
        elif sm == "squared":
            # For numeric sequences, use squared difference
            sm_matrix = np.zeros((n_states, n_states), dtype=np.float64)
            for i in range(n_states):
                for j in range(n_states):
                    sm_matrix[i, j] = (i - j) ** 2
        else:
            raise ValueError(f"Unknown substitution method: {sm}")
    else:
        sm_matrix = sm.astype(np.float64)

    # Compute pairwise distance matrix
    dist_matrix = _compute_pairwise_distances(seq_a, seq_b, sm_matrix)

    # Compute SoftDTW
    if window > 0:
        distance = _softdtw_kernel_with_window(dist_matrix, gamma, window)
    else:
        distance = _softdtw_kernel(dist_matrix, gamma)

    if normalize:
        total_len = len(seq_a) + len(seq_b)
        if total_len > 0:
            distance /= total_len

    return float(distance)


def softdtw_divergence(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    gamma: float = 1.0,
    window: int = 0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 1.0,
) -> float:
    """
    Compute SoftDTW divergence between two sequences.

    The divergence is defined as:
        D(a, b) = SoftDTW(a, b) - 0.5 * (SoftDTW(a, a) + SoftDTW(b, b))

    This makes the divergence 0 when the sequences are identical and
    provides a more symmetric measure.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        gamma: Smoothing parameter (> 0).
        window: Sakoe-Chiba band width (0 = no constraint).
        sm: Substitution cost matrix or method.
        sub_cost: Cost for non-matching elements.

    Returns:
        SoftDTW divergence.

    Example:
        >>> import numpy as np
        >>> seq1 = np.array([0, 1, 2])
        >>> seq2 = np.array([0, 1, 2])
        >>> softdtw_divergence(seq1, seq2, gamma=1.0)  # Close to 0
    """
    sdtw_ab = softdtw_distance(seq_a, seq_b, gamma, window, sm, sub_cost)
    sdtw_aa = softdtw_distance(seq_a, seq_a, gamma, window, sm, sub_cost)
    sdtw_bb = softdtw_distance(seq_b, seq_b, gamma, window, sm, sub_cost)

    return sdtw_ab - 0.5 * (sdtw_aa + sdtw_bb)


class SoftDTWMetric:
    """Soft-DTW distance metric class."""

    name = "softdtw"

    def __init__(
        self,
        gamma: float = 1.0,
        window: int = 0,
        sm: np.ndarray | str = "constant",
        sub_cost: float = 1.0,
        normalize: bool = False,
        use_divergence: bool = False,
    ) -> None:
        """
        Initialize the SoftDTW metric.

        Args:
            gamma: Smoothing parameter (> 0).
            window: Sakoe-Chiba band width (0 = no constraint).
            sm: Substitution cost matrix or method.
            sub_cost: Cost for non-matching elements.
            normalize: Whether to normalize by sequence lengths.
            use_divergence: If True, compute SoftDTW divergence instead.
        """
        self.gamma = gamma
        self.window = window
        self.sm = sm
        self.sub_cost = sub_cost
        self.normalize = normalize
        self.use_divergence = use_divergence

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """
        Compute SoftDTW distance or divergence between two sequences.

        Args:
            seq_a: First sequence (integer-encoded).
            seq_b: Second sequence (integer-encoded).

        Returns:
            SoftDTW distance or divergence.
        """
        if self.use_divergence:
            return softdtw_divergence(
                seq_a,
                seq_b,
                gamma=self.gamma,
                window=self.window,
                sm=self.sm,
                sub_cost=self.sub_cost,
            )
        else:
            return softdtw_distance(
                seq_a,
                seq_b,
                gamma=self.gamma,
                window=self.window,
                sm=self.sm,
                sub_cost=self.sub_cost,
                normalize=self.normalize,
            )
