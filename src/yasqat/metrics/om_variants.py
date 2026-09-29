"""yasqat's own Optimal Matching heuristics (not TraMineR's OM variants).

- ``om_boundary_weighted_distance``: substitution costs weighted by distance
  from the sequence ends.
- ``om_spell_scaled_distance``: substitution costs scaled down inside long
  spells.
- ``om_transition_penalty_distance``: an added penalty when the transition
  leading into the two positions differs.

These were shipped through 0.5.0 under the names ``omloc``, ``omspell`` and
``omstran``. They are **not** TraMineR's OMloc, OMspell or OMstran and were
renamed so that they no longer suggest they are. TraMineR's methods now live
in :mod:`yasqat.metrics.omloc`, :mod:`yasqat.metrics.omspell` and
:mod:`yasqat.metrics.omstran` under the names TraMineR uses.
"""

from __future__ import annotations

import numba
import numpy as np

# ---------- Boundary-weighted OM ----------


@numba.jit(nopython=True, cache=True, nogil=True)
def _om_boundary_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    context_factor: float,
) -> float:
    """
    Numba kernel: substitution costs weighted by distance from the ends.

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


def om_boundary_weighted_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    context_factor: float = 1.0,
    normalize: bool = False,
) -> float:
    """
    Compute a boundary-weighted Optimal Matching distance.

    Substitution costs are scaled by ``1 + context_factor * (1 - p)`` where
    ``p`` is the position's relative distance from the nearest sequence end,
    so edits near the boundaries cost more than edits in the middle.

    A yasqat heuristic, formerly named ``omloc``. It is **not** TraMineR's
    OMloc; see the module docstring.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost.
        sm: Substitution cost matrix or "constant".
        sub_cost: Constant substitution cost (when sm="constant").
        context_factor: How much position affects cost (0 = standard OM).
        normalize: If True, normalize by max sequence length.

    Returns:
        Boundary-weighted OM distance.
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

    distance = _om_boundary_kernel(seq_a, seq_b, indel, sm_matrix, context_factor)

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)


# ---------- Spell-scaled OM ----------


@numba.jit(nopython=True, cache=True, nogil=True)
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


@numba.jit(nopython=True, cache=True, nogil=True)
def _om_spell_scaled_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    spell_lens_a: np.ndarray,
    spell_lens_b: np.ndarray,
) -> float:
    """
    Numba kernel: substitution costs scaled by spell length.

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


def om_spell_scaled_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    normalize: bool = False,
) -> float:
    """
    Compute a spell-scaled Optimal Matching distance.

    Each substitution cost is multiplied by
    ``1 / sqrt(spell_len_a * spell_len_b)``, the lengths of the spells the two
    positions belong to, so edits inside long stable spells cost less.

    A yasqat heuristic, formerly named ``omspell``. It is **not** TraMineR's
    OMspell; see the module docstring.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Insertion/deletion cost.
        sm: Substitution cost matrix or "constant".
        sub_cost: Constant substitution cost (when sm="constant").
        normalize: If True, normalize by max sequence length.

    Returns:
        Spell-scaled OM distance.
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

    distance = _om_spell_scaled_kernel(
        seq_a, seq_b, indel, sm_matrix, spell_lens_a, spell_lens_b
    )

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)


# ---------- Transition-penalty OM ----------


@numba.jit(nopython=True, cache=True, nogil=True)
def _om_transition_penalty_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel_cost: float,
    sub_costs: np.ndarray,
    transition_weights: np.ndarray,
    otto: float,
) -> float:
    """
    Numba kernel: substitution cost plus a transition-difference penalty.

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


def om_transition_penalty_distance(
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
    Compute a transition-penalty Optimal Matching distance.

    On top of the state substitution cost, a substitution pays
    ``otto * |w_a - w_b|`` where ``w_a`` and ``w_b`` are the weights of the
    transitions that led into the two positions.

    A yasqat heuristic, formerly named ``omstran``. It is **not** TraMineR's
    OMstran; see the module docstring.

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
        Transition-penalty OM distance.
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

    distance = _om_transition_penalty_kernel(
        seq_a, seq_b, indel, sm_matrix, transition_weights, otto
    )

    if normalize:
        max_len = max(len(seq_a), len(seq_b))
        if max_len > 0:
            distance /= max_len

    return float(distance)
