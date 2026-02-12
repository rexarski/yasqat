"""Non-metric sequence metrics: NMS, NMSMST, SVRspell.

NMS: Number of matching subsequences.
NMSMST: NMS with minimum spanning tree normalization.
SVRspell: State-visit representativeness based on spell structure.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _count_common_subsequences(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> int:
    """Count the number of common subsequences between two sequences.

    Uses DP: dp[i][j] = number of common subsequences of seq_a[:i] and seq_b[:j].
    dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]          if a[i] != b[j]
    dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + dp[i-1][j-1] + 1  if a[i] == b[j]
             = dp[i-1][j] + dp[i][j-1] + 1                                   if a[i] == b[j]
    """
    n = len(seq_a)
    m = len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = dp[i - 1, j] + dp[i, j - 1] - dp[i - 1, j - 1]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i, j] += dp[i - 1, j - 1] + 1

    return int(dp[n, m])


def nms_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute NMS (Number of Matching Subsequences) distance.

    NMS counts the number of common subsequences and converts to distance.
    Higher NMS means more similar sequences, so distance = 1 / (1 + NMS)
    or normalized variant.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        normalize: If True, normalize to [0, 1] range.

    Returns:
        NMS-based distance (0 = identical, higher = more different).
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 1.0

    nms = _count_common_subsequences(seq_a, seq_b)
    # Self-similarities for normalization
    nms_aa = _count_common_subsequences(seq_a, seq_a)
    nms_bb = _count_common_subsequences(seq_b, seq_b)

    if normalize:
        max_nms = max(nms_aa, nms_bb)
        if max_nms > 0:
            return 1.0 - (nms / max_nms)
        return 0.0

    return float(max(0, max(nms_aa, nms_bb) - nms))


def nmsmst_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """
    Compute NMSMST distance (NMS with minimum spanning tree normalization).

    Uses geometric mean of self-similarities for normalization:
    d = 1 - NMS(a,b) / sqrt(NMS(a,a) * NMS(b,b))

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).

    Returns:
        NMSMST distance (0 = identical, 1 = completely different).
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 1.0

    nms = _count_common_subsequences(seq_a, seq_b)
    nms_aa = _count_common_subsequences(seq_a, seq_a)
    nms_bb = _count_common_subsequences(seq_b, seq_b)

    denom = np.sqrt(float(nms_aa) * float(nms_bb))
    if denom > 0:
        return max(0.0, 1.0 - nms / denom)
    return 0.0


@numba.jit(nopython=True, cache=True)
def _spell_vectors(seq: np.ndarray, n_states: int) -> np.ndarray:
    """Extract spell structure: for each state, sorted list of spell lengths."""
    # Count spells per state
    max_spells = len(seq)
    # spell_info[state] = list of spell lengths
    counts = np.zeros((n_states, max_spells), dtype=np.int64)
    n_spells = np.zeros(n_states, dtype=np.int64)

    if len(seq) == 0:
        return counts

    current_state = seq[0]
    current_len = 1
    for i in range(1, len(seq)):
        if seq[i] == current_state:
            current_len += 1
        else:
            idx = n_spells[current_state]
            counts[current_state, idx] = current_len
            n_spells[current_state] += 1
            current_state = seq[i]
            current_len = 1

    # Final spell
    idx = n_spells[current_state]
    counts[current_state, idx] = current_len
    n_spells[current_state] += 1

    return counts


def svrspell_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    n_states: int | None = None,
) -> float:
    """
    Compute SVRspell distance (State Visit Representativeness based on spells).

    Compares sequences based on their spell structure: which states are visited
    and how long each spell lasts. Uses L1 distance on sorted spell-length
    vectors for each state.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        n_states: Number of states. If None, inferred from data.

    Returns:
        SVRspell distance (0 = identical spell structure).
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float(max(len(seq_a), len(seq_b)))

    if n_states is None:
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1

    spells_a = _spell_vectors(seq_a, n_states)
    spells_b = _spell_vectors(seq_b, n_states)

    total_dist = 0.0
    for s in range(n_states):
        # Get non-zero spell lengths and sort
        lens_a = sorted([int(x) for x in spells_a[s] if x > 0], reverse=True)
        lens_b = sorted([int(x) for x in spells_b[s] if x > 0], reverse=True)

        # Pad shorter list with zeros
        max_len = max(len(lens_a), len(lens_b))
        while len(lens_a) < max_len:
            lens_a.append(0)
        while len(lens_b) < max_len:
            lens_b.append(0)

        # L1 distance on sorted spell lengths
        for i in range(max_len):
            total_dist += abs(lens_a[i] - lens_b[i])

    return total_dist
