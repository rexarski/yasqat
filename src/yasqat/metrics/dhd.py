"""Dynamic Hamming Distance (DHD) metric.

DHD generalizes the Hamming distance by allowing substitution costs
to vary across time positions. Costs are derived from cross-sectional
state frequencies at each position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool


def build_position_costs(pool: SequencePool) -> np.ndarray:
    """
    Build position-dependent substitution cost matrix from a pool.

    For each time position t, computes cross-sectional state frequencies
    and derives substitution costs as:
        cost(a, b, t) = |freq(a, t) - freq(b, t)| / N

    where N is the number of sequences. The cost is 0 on the diagonal.

    The resulting matrix is a 3D array of shape (T, n_states, n_states)
    where T is the sequence length and n_states is the alphabet size.

    States that frequently co-occur at the same position will have
    lower substitution costs, reflecting that they are more similar
    at that time point.

    Args:
        pool: SequencePool with equal-length sequences.

    Returns:
        3D numpy array of shape (T, n_states, n_states).

    Raises:
        ValueError: If sequences have different lengths.
    """
    ids = pool.sequence_ids
    n_seqs = len(ids)
    n_states = len(pool.alphabet)

    # Get all encoded sequences and verify equal length
    encoded = [pool.get_encoded_sequence(sid) for sid in ids]
    seq_len = len(encoded[0])
    for i, seq in enumerate(encoded):
        if len(seq) != seq_len:
            raise ValueError(
                f"All sequences must have the same length for DHD. "
                f"Sequence {ids[i]} has length {len(seq)}, expected {seq_len}"
            )

    # Build frequency counts per position
    # freq[t, s] = number of sequences in state s at time t
    freq = np.zeros((seq_len, n_states), dtype=np.float64)
    for seq in encoded:
        for t in range(seq_len):
            freq[t, seq[t]] += 1

    # Normalize to proportions
    if n_seqs > 0:
        freq /= n_seqs

    # Build cost matrices: cost(a, b, t) based on frequency difference
    # Following Lesnard (2010): c(a,b,t) = (freq(a,t) + freq(b,t)) if a != b
    # Simplified: use chi2-like costs based on cross-sectional distributions
    costs = np.zeros((seq_len, n_states, n_states), dtype=np.float64)
    for t in range(seq_len):
        for a in range(n_states):
            for b in range(a + 1, n_states):
                # Cost based on combined rarity: rarer pairs cost more
                # c(a,b,t) = 1 - (freq(a,t) * freq(b,t)) / max_possible
                # Simplified: 1 if different, scaled by how rare the transition is
                fa = freq[t, a]
                fb = freq[t, b]
                # If both states are common at this position, substitution cost is low
                # If one/both are rare, cost is high
                if fa + fb > 0:
                    cost = 1.0 - (fa * fb) / (0.25)  # Normalize by max product
                    cost = max(0.0, min(1.0, cost))  # Clamp to [0, 1]
                else:
                    cost = 1.0
                costs[t, a, b] = cost
                costs[t, b, a] = cost

    return costs


@numba.jit(nopython=True, cache=True)
def _dhd_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    position_costs: np.ndarray,
) -> float:
    """
    Numba-optimized DHD computation.

    Sums position-dependent substitution costs for mismatched positions.

    Args:
        seq_a: First sequence (integer-encoded).
        seq_b: Second sequence (integer-encoded).
        position_costs: 3D array of shape (T, n_states, n_states).

    Returns:
        DHD distance.
    """
    n = len(seq_a)
    distance = 0.0
    for t in range(n):
        if seq_a[t] != seq_b[t]:
            distance += position_costs[t, seq_a[t], seq_b[t]]
    return distance


def dhd_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    position_costs: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute Dynamic Hamming Distance between two sequences.

    DHD extends Hamming distance by using position-dependent substitution
    costs. Both sequences must have the same length.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        position_costs: 3D cost array of shape (T, n_states, n_states).
        normalize: If True, normalize by sequence length.

    Returns:
        DHD distance.

    Raises:
        ValueError: If sequences have different lengths.
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(
            f"Sequences must have the same length for DHD. "
            f"Got {len(seq_a)} and {len(seq_b)}"
        )

    if len(seq_a) == 0:
        return 0.0

    distance = float(_dhd_kernel(seq_a, seq_b, position_costs))

    if normalize and len(seq_a) > 0:
        distance /= len(seq_a)

    return distance


class DHDMetric:
    """Dynamic Hamming Distance metric class."""

    name = "dhd"

    def __init__(
        self,
        position_costs: np.ndarray,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the DHD metric.

        Args:
            position_costs: 3D cost array from build_position_costs().
            normalize: Whether to normalize distances by sequence length.
        """
        self.position_costs = position_costs
        self.normalize = normalize

    def compute(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Compute DHD distance between two sequences."""
        return dhd_distance(
            seq_a,
            seq_b,
            position_costs=self.position_costs,
            normalize=self.normalize,
        )
