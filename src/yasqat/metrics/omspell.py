"""Spell-sensitive Optimal Matching (TraMineR ``seqdist(method="OMspell")``).

Transcribed from TraMineR 2.2-14: ``R/seqdist.R`` (spell preparation) and
``src/OMPerdistance.cpp``. Sequences are first collapsed into spells
``(state, duration)``; each spell carries ``d' = duration ** tpow - 1``.
Over the spell sequences the costs are (``timecost = expcost``):

    indel(spell)            = indel[state] + timecost * d'
    sub(same state)         = timecost * |d'_a - d'_b|
    sub(different states)   = sm[a, b] + timecost * (d'_a + d'_b)

Normalisation uses the original (time-point) lengths.
"""

from __future__ import annotations

import numba
import numpy as np

from yasqat.metrics._normalize import max_sub_cost, normalize_distance, resolve_norm


@numba.jit(nopython=True, cache=True, nogil=True)
def _to_spells(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode ``seq`` into (states, durations)."""
    n = len(seq)
    states = np.empty(n, dtype=np.int64)
    durations = np.empty(n, dtype=np.float64)
    k = 0
    for i in range(n):
        if i > 0 and seq[i] == seq[i - 1]:
            durations[k - 1] += 1.0
        else:
            states[k] = seq[i]
            durations[k] = 1.0
            k += 1
    return states[:k], durations[:k]


@numba.jit(nopython=True, cache=True, nogil=True)
def _omspell_kernel(
    st_a: np.ndarray,
    dur_a: np.ndarray,
    st_b: np.ndarray,
    dur_b: np.ndarray,
    indels: np.ndarray,
    sm: np.ndarray,
    timecost: float,
) -> float:
    """``OMPerdistance::distance``; ``dur_*`` already hold ``d ** tpow - 1``."""
    m = len(st_a)
    n = len(st_b)
    fmat = np.zeros((m + 1, n + 1), dtype=np.float64)
    for ii in range(1, m + 1):
        fmat[ii, 0] = fmat[ii - 1, 0] + indels[st_a[ii - 1]] + timecost * dur_a[ii - 1]
    for jj in range(1, n + 1):
        fmat[0, jj] = fmat[0, jj - 1] + indels[st_b[jj - 1]] + timecost * dur_b[jj - 1]

    for j in range(1, n + 1):
        sj = st_b[j - 1]
        dj = dur_b[j - 1]
        for i in range(1, m + 1):
            si = st_a[i - 1]
            di = dur_a[i - 1]
            best = fmat[i, j - 1] + indels[sj] + timecost * dj
            other = fmat[i - 1, j] + indels[si] + timecost * di
            if other < best:
                best = other
            if si == sj:
                sub = fmat[i - 1, j - 1] + timecost * abs(di - dj)
            else:
                sub = fmat[i - 1, j - 1] + sm[si, sj] + (di + dj) * timecost
            fmat[i, j] = sub if sub < best else best
    return fmat[m, n]


def omspell_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    indel: float | np.ndarray = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    expcost: float = 0.5,
    tpow: float = 1.0,
    normalize: bool | str = False,
) -> float:
    """
    Compute TraMineR's spell-sensitive OM distance (``seqdist(method="OMspell")``).

    The sequences are compared as sequences of spells. Two spells of the same
    state differ only by ``expcost`` times their duration difference; spells
    of different states cost the substitution cost plus ``expcost`` times the
    sum of their (transformed) durations.

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        indel: Indel cost, a scalar or one value per state. TraMineR's
            ``"auto"`` would be ``max(sm) / 2``.
        sm: Substitution cost matrix, or ``"constant"``.
        sub_cost: Constant substitution cost when ``sm="constant"``.
        expcost: Duration cost factor (``timecost``), ``>= 0``. Default 0.5.
        tpow: Exponent applied to durations before the ``- 1`` shift.
        normalize: ``False``, ``True`` / ``"auto"`` (``"yujianbo"`` for
            OMspell), or an explicit TraMineR norm name.

    Returns:
        OMspell distance.
    """
    if expcost < 0:
        raise ValueError("'expcost' must be positive")
    m_len, n_len = len(seq_a), len(seq_b)
    if m_len == 0 and n_len == 0:
        return 0.0

    n_states = (
        max(int(seq_a.max()) if m_len else -1, int(seq_b.max()) if n_len else -1) + 1
    )
    if isinstance(sm, str):
        if sm != "constant":
            raise ValueError(f"Unknown substitution method: {sm}")
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
    else:
        sm_matrix = np.asarray(sm, dtype=np.float64)
        if sm_matrix.ndim != 2 or sm_matrix.shape[0] != sm_matrix.shape[1]:
            raise ValueError(
                f"Substitution matrix must be square; got shape {sm_matrix.shape}."
            )
        if sm_matrix.shape[0] < n_states:
            raise ValueError(
                f"Substitution matrix has shape {sm_matrix.shape} but the sequences "
                f"contain state index {n_states - 1}."
            )
        n_states = sm_matrix.shape[0]
    if np.isscalar(indel):
        indels = np.full(n_states, float(indel), dtype=np.float64)  # type: ignore[arg-type]
    else:
        indels = np.asarray(indel, dtype=np.float64)
        if indels.shape != (n_states,):
            raise ValueError(
                f"'indel' must be a scalar or have one value per state ({n_states})."
            )

    st_a, dur_a = _to_spells(seq_a)
    st_b, dur_b = _to_spells(seq_b)
    dur_a = dur_a**tpow - 1.0
    dur_b = dur_b**tpow - 1.0

    norm = resolve_norm(normalize, auto="yujianbo")
    indel_max = float(indels.max())
    maxscost = max_sub_cost(sm_matrix, indel_max, norm)
    raw = _omspell_kernel(st_a, dur_a, st_b, dur_b, indels, sm_matrix, expcost)
    maxdist = abs(n_len - m_len) * indel_max + maxscost * min(m_len, n_len)
    return float(
        normalize_distance(raw, maxdist, m_len * indel_max, n_len * indel_max, norm)
    )
