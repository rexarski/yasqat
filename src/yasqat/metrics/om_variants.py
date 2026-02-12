"""Optimal Matching variants: OMloc, OMspell, OMstran.

OMloc: Localized OM with position-dependent substitution costs.
OMspell: Spell-length sensitive OM that penalizes changes within long spells.
OMstran: Transition-sensitive OM that accounts for transition frequencies.
"""

from __future__ import annotations

import numba
import numpy as np

# ---------- OMloc (Localized OM) ----------


@numba.jit(nopython=True, cache=True)
def _omloc_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    context_factor: float,
) -> float:
    """
    Numba-optimized OMloc: position-dependent substitution costs.

    Substitution cost at position t is weighted by a context factor based
    on how far the position is from the nearest sequence boundary.
    """
    n = len(seq_a)
    m = len(seq_b)

    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    for i in range(n + 1):
        dp[i, 0] = i * indel_cost
    for j in range(m + 1):
        dp[0, j] = j * indel_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Position weight: higher at boundaries, lower in center
            pos_a = min(i - 1, n - i) / max(n - 1, 1)
            pos_b = min(j - 1, m - j) / max(m - 1, 1)
            local_weight = 1.0 + context_factor * (1.0 - (pos_a + pos_b) / 2.0)

            sub_cost = sub_costs[seq_a[i - 1], seq_b[j - 1]] * local_weight

            dp[i, j] = min(
                dp[i - 1, j] + indel_cost,
                dp[i, j - 1] + indel_cost,
                dp[i - 1, j - 1] + sub_cost,
            )

    return dp[n, m]


def omloc_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    context_factor: float = 1.0,
    normalize: bool = False,
) -> float:
    """
    Compute Localized Optimal Matching distance.

    OMloc weights substitution costs by position, giving more importance
    to boundary positions. This is useful when the beginning and end of
    sequences carry more structural meaning.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost.
        sm: Substitution cost matrix or "constant".
        sub_cost: Constant substitution cost (when sm="constant").
        context_factor: How much position affects cost (0 = standard OM).
        normalize: If True, normalize by max sequence length.

    Returns:
        OMloc distance.
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0:
        return float(len(seq_b) * indel)
    if len(seq_b) == 0:
        return float(len(seq_a) * indel)

    if isinstance(sm, str):
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
    else:
        sm_matrix = sm.astype(np.float64)

    distance = _omloc_kernel(seq_a, seq_b, indel, sm_matrix, context_factor)

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)


# ---------- OMspell (Spell-length sensitive OM) ----------


@numba.jit(nopython=True, cache=True)
def _compute_spell_lengths(seq: np.ndarray) -> np.ndarray:
    """Compute the spell length at each position."""
    n = len(seq)
    spell_lens = np.ones(n, dtype=np.float64)

    # Forward pass: mark spell lengths
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            spell_lens[i] = spell_lens[i - 1] + 1

    # Backward pass: propagate max spell length to start of spell
    max_spell = spell_lens[n - 1]
    for i in range(n - 2, -1, -1):
        if seq[i] == seq[i + 1]:
            spell_lens[i] = max_spell
        else:
            max_spell = spell_lens[i]

    return spell_lens


@numba.jit(nopython=True, cache=True)
def _omspell_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    spell_lens_a: np.ndarray,
    spell_lens_b: np.ndarray,
) -> float:
    """
    Numba-optimized OMspell: spell-length weighted substitution costs.

    Substitutions within longer spells are penalized more, as they
    represent more significant structural changes.
    """
    n = len(seq_a)
    m = len(seq_b)

    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    for i in range(n + 1):
        dp[i, 0] = i * indel_cost
    for j in range(m + 1):
        dp[0, j] = j * indel_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Weight by inverse of geometric mean of spell lengths
            weight = 1.0 / np.sqrt(spell_lens_a[i - 1] * spell_lens_b[j - 1])
            sub = sub_costs[seq_a[i - 1], seq_b[j - 1]] * weight

            dp[i, j] = min(
                dp[i - 1, j] + indel_cost,
                dp[i, j - 1] + indel_cost,
                dp[i - 1, j - 1] + sub,
            )

    return dp[n, m]


def omspell_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    normalize: bool = False,
) -> float:
    """
    Compute Spell-length sensitive Optimal Matching distance.

    OMspell reduces the substitution cost within long spells, reflecting
    that changes within stable periods are less meaningful than changes
    at transitions.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost.
        sm: Substitution cost matrix or "constant".
        sub_cost: Constant substitution cost (when sm="constant").
        normalize: If True, normalize by max sequence length.

    Returns:
        OMspell distance.
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0:
        return float(len(seq_b) * indel)
    if len(seq_b) == 0:
        return float(len(seq_a) * indel)

    if isinstance(sm, str):
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
    else:
        sm_matrix = sm.astype(np.float64)

    spell_lens_a = _compute_spell_lengths(seq_a)
    spell_lens_b = _compute_spell_lengths(seq_b)

    distance = _omspell_kernel(
        seq_a, seq_b, indel, sm_matrix, spell_lens_a, spell_lens_b
    )

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)


# ---------- OMstran (Transition-sensitive OM) ----------


@numba.jit(nopython=True, cache=True)
def _omstran_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    transition_weights: np.ndarray,
    otto: float,
) -> float:
    """
    Numba-optimized OMstran: transition-sensitive substitution costs.

    Adds extra cost when a substitution changes the transition context.
    """
    n = len(seq_a)
    m = len(seq_b)

    dp = np.zeros((n + 1, m + 1), dtype=np.float64)

    for i in range(n + 1):
        dp[i, 0] = i * indel_cost
    for j in range(m + 1):
        dp[0, j] = j * indel_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            base_sub = sub_costs[seq_a[i - 1], seq_b[j - 1]]

            # Add transition context cost
            trans_cost = 0.0
            if i > 1 and j > 1:
                # Cost of changing the transition s[i-2]->s[i-1] vs s[j-2]->s[j-1]
                ta = transition_weights[seq_a[i - 2], seq_a[i - 1]]
                tb = transition_weights[seq_b[j - 2], seq_b[j - 1]]
                trans_cost = otto * abs(ta - tb)

            dp[i, j] = min(
                dp[i - 1, j] + indel_cost,
                dp[i, j - 1] + indel_cost,
                dp[i - 1, j - 1] + base_sub + trans_cost,
            )

    return dp[n, m]


def omstran_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    transition_weights: np.ndarray | None = None,
    otto: float = 1.0,
    normalize: bool = False,
) -> float:
    """
    Compute Transition-sensitive Optimal Matching distance.

    OMstran incorporates transition frequency information: substitutions
    that change a common transition pattern are penalized more.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost.
        sm: Substitution cost matrix or "constant".
        sub_cost: Constant substitution cost (when sm="constant").
        transition_weights: Matrix of transition frequencies/weights.
            If None, uses identity (transitions to same state = 1, else 0).
        otto: Weight for the transition cost component.
        normalize: If True, normalize by max sequence length.

    Returns:
        OMstran distance.
    """
    if len(seq_a) == 0 and len(seq_b) == 0:
        return 0.0
    if len(seq_a) == 0:
        return float(len(seq_b) * indel)
    if len(seq_b) == 0:
        return float(len(seq_a) * indel)

    if isinstance(sm, str):
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
    else:
        sm_matrix = sm.astype(np.float64)
        n_states = sm_matrix.shape[0]

    if transition_weights is None:
        transition_weights = np.eye(n_states, dtype=np.float64)
    else:
        transition_weights = transition_weights.astype(np.float64)

    distance = _omstran_kernel(seq_a, seq_b, indel, sm_matrix, transition_weights, otto)

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)
