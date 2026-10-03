"""Localized Optimal Matching (TraMineR ``seqdist(method="OMloc")``).

Transcribed from TraMineR 2.2-14: ``R/seqdist.R`` (parameter preparation)
and ``src/OMVIdistance.cpp`` with ``OMlocIndelCalculator``. OMloc is
Hollister's (2009) variant in which the cost of inserting or deleting a
state depends on the states that surround the position in the *other*
sequence:

    indel(x | prev, next) = timecost + localcost * (sm[prev, x] + sm[next, x]) / 2

with ``timecost = expcost * maxscost`` and ``localcost = context``. Substitution
costs come from ``sm`` unchanged.
"""

from __future__ import annotations

import numba
import numpy as np

from yasqat.metrics._normalize import max_sub_cost, normalize_distance, resolve_norm


@numba.jit(nopython=True, cache=True, nogil=True)
def _omloc_kernel(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sm: np.ndarray,
    timecost: float,
    localcost: float,
) -> float:
    """``OMVIdistance::distance`` with ``indelmethod == 1``."""
    m = len(seq_a)
    n = len(seq_b)
    fmat = np.zeros((m + 1, n + 1), dtype=np.float64)

    # Border: the context for the first column/row is the other sequence's
    # first state, used as both ``prev`` and ``next`` (firststate == 0).
    b0 = seq_b[0]
    for ii in range(1, m + 1):
        x = seq_a[ii - 1]
        fmat[ii, 0] = (
            fmat[ii - 1, 0] + timecost + localcost * (sm[b0, x] + sm[b0, x]) / 2.0
        )
    a0 = seq_a[0]
    for jj in range(1, n + 1):
        x = seq_b[jj - 1]
        fmat[0, jj] = (
            fmat[0, jj - 1] + timecost + localcost * (sm[a0, x] + sm[a0, x]) / 2.0
        )

    prev_j = seq_b[0]
    for j in range(1, n + 1):
        j_state = seq_b[j - 1]
        prev_i = seq_a[0]
        for i in range(1, m + 1):
            i_state = seq_a[i - 1]
            # Delete seq_a[i-1] in the context of seq_b around j.
            del_cost = (
                timecost
                + localcost * (sm[prev_j, i_state] + sm[j_state, i_state]) / 2.0
            )
            best = fmat[i - 1, j] + del_cost
            # Insert seq_b[j-1] in the context of seq_a around i.
            ins_cost = (
                timecost
                + localcost * (sm[prev_i, j_state] + sm[i_state, j_state]) / 2.0
            )
            ins = fmat[i, j - 1] + ins_cost
            if ins < best:
                best = ins
            if i_state == j_state:
                sub = fmat[i - 1, j - 1]
            else:
                sub = fmat[i - 1, j - 1] + sm[i_state, j_state]
            fmat[i, j] = sub if sub < best else best
            prev_i = i_state
        prev_j = j_state
    return fmat[m, n]


def omloc_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    expcost: float = 0.5,
    context: float | None = None,
    normalize: bool | str = False,
) -> float:
    """
    Compute TraMineR's localized OM distance (``seqdist(method="OMloc")``).

    The indel cost of a state depends on the states around the position in
    the other sequence, so inserting a state that resembles its neighbours
    is cheap. With the defaults (``expcost=0.5``, ``context=0``) every indel
    costs ``max(sm) / 2`` and the result equals plain OM with that indel.

    Args:
        seq_a: First sequence (integer-encoded numpy array, non-empty).
        seq_b: Second sequence (integer-encoded numpy array, non-empty).
        sm: Substitution cost matrix, or ``"constant"`` for a matrix filled
            with ``sub_cost`` off the diagonal.
        sub_cost: Constant substitution cost when ``sm="constant"``.
        expcost: Time-cost factor, ``>= 0``. TraMineR default 0.5.
        context: Local-context factor, ``>= 0``. Defaults to
            ``1 - 2 * expcost`` as in TraMineR, which requires
            ``expcost <= 0.5`` unless ``context`` is given explicitly.
        normalize: ``False`` (raw), ``True`` / ``"auto"`` (TraMineR's choice
            for OMloc: ``"yujianbo"``), or one of ``"maxlength"``,
            ``"gmean"``, ``"maxdist"``, ``"yujianbo"``.

    Returns:
        OMloc distance.
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        raise ValueError(
            "OMloc is not defined for empty sequences (TraMineR stops too)."
        )
    if expcost < 0:
        raise ValueError("'expcost' must be positive")
    if context is None:
        context = 1.0 - 2.0 * expcost
    if context < 0:
        raise ValueError("'context' must be positive ('expcost' must be in [0, 0.5])")

    sm_matrix = _resolve_sm(seq_a, seq_b, sm, sub_cost)
    norm = resolve_norm(normalize, auto="yujianbo")
    # seqdist.R: params$indel <- max(sm) * expcost + context (used for
    # normalisation and for capping maxscost).
    indel = float(sm_matrix.max()) * expcost + context
    maxscost = max_sub_cost(sm_matrix, indel, norm)
    timecost = expcost * maxscost
    raw = _omloc_kernel(seq_a, seq_b, sm_matrix, timecost, context)

    m, n = len(seq_a), len(seq_b)
    maxdist = abs(n - m) * indel + maxscost * min(m, n)
    return float(normalize_distance(raw, maxdist, m * indel, n * indel, norm))


def _resolve_sm(
    seq_a: np.ndarray, seq_b: np.ndarray, sm: np.ndarray | str, sub_cost: float
) -> np.ndarray:
    if isinstance(sm, str):
        if sm != "constant":
            raise ValueError(f"Unknown substitution method: {sm}")
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
        return sm_matrix
    sm_matrix = np.asarray(sm, dtype=np.float64)
    if sm_matrix.ndim != 2 or sm_matrix.shape[0] != sm_matrix.shape[1]:
        raise ValueError(
            f"Substitution matrix must be square; got shape {sm_matrix.shape}."
        )
    required = max(int(seq_a.max()), int(seq_b.max())) + 1
    if sm_matrix.shape[0] < required:
        raise ValueError(
            f"Substitution matrix has shape {sm_matrix.shape} but the sequences "
            f"contain state index {required - 1}."
        )
    return sm_matrix
