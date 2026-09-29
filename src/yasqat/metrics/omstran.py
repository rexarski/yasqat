"""Transition-sensitive Optimal Matching (TraMineR ``seqdist(method="OMstran")``).

Transcribed from TraMineR 2.2-14 ``R/seqdist-OMstran.R``. Each sequence is
rewritten as a sequence of transitions ``(s_t, s_{t+1})`` (the last position
pairs with itself, TraMineR's ``add.column=TRUE``); a substitution matrix and
per-transition indel costs are derived from the state costs, weighting the
origin state by ``otto`` and the transition by ``1 - otto``; then plain OM
with per-token indels runs over the transition sequences
(``OMVIdistance`` with ``indelmethod == 0``).

Only TraMineR's defaults ``previous=FALSE`` and ``add.column=TRUE`` are
implemented.
"""

from __future__ import annotations

import numba
import numpy as np

from yasqat.metrics._normalize import max_sub_cost, normalize_distance, resolve_norm


@numba.jit(nopython=True, cache=True, nogil=True)
def _om_varying_indel_kernel(
    seq_a: np.ndarray, seq_b: np.ndarray, indels: np.ndarray, sm: np.ndarray
) -> float:
    """OM with a per-state indel vector (``VaryingIndelCalculator``)."""
    m = len(seq_a)
    n = len(seq_b)
    fmat = np.zeros((m + 1, n + 1), dtype=np.float64)
    for ii in range(1, m + 1):
        fmat[ii, 0] = fmat[ii - 1, 0] + indels[seq_a[ii - 1]]
    for jj in range(1, n + 1):
        fmat[0, jj] = fmat[0, jj - 1] + indels[seq_b[jj - 1]]
    for j in range(1, n + 1):
        sj = seq_b[j - 1]
        for i in range(1, m + 1):
            si = seq_a[i - 1]
            best = fmat[i - 1, j] + indels[si]
            other = fmat[i, j - 1] + indels[sj]
            if other < best:
                best = other
            sub = fmat[i - 1, j - 1] + (0.0 if si == sj else sm[si, sj])
            fmat[i, j] = sub if sub < best else best
    return fmat[m, n]


def _transition_tokens(seq: np.ndarray, n_states: int) -> np.ndarray:
    """``(s_t, s_{t+1})`` encoded as ``s_t * n_states + s_{t+1}``; last pairs with itself."""
    nxt = np.empty_like(seq)
    nxt[:-1] = seq[1:]
    nxt[-1] = seq[-1]
    return (seq.astype(np.int64) * n_states + nxt.astype(np.int64)).astype(np.int64)


def omstran_costs(
    n_states: int,
    sm: np.ndarray,
    indel: float | np.ndarray,
    otto: float,
    transindel: str = "constant",
    transition_rates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the transition-token indel vector and substitution matrix.

    Returns ``(indels, newsm)`` over the ``n_states ** 2`` tokens
    ``a * n_states + b``, exactly as ``seqdist-OMstran.R`` derives them.
    """
    sm = np.asarray(sm, dtype=np.float64)
    if np.isscalar(indel):
        indel_vec = np.full(n_states, float(indel), dtype=np.float64)  # type: ignore[arg-type]
    else:
        indel_vec = np.asarray(indel, dtype=np.float64)
    if transindel not in ("constant", "prob", "subcost"):
        raise ValueError("'transindel' must be one of 'constant', 'prob', 'subcost'")
    if transindel == "prob":
        if transition_rates is None:
            raise ValueError("transindel='prob' requires 'transition_rates'")
        tr = np.asarray(transition_rates, dtype=np.float64)

    sm_max = float(sm.max())
    indel_max = float(indel_vec.max())
    transweight = 1.0 - otto
    indelrate = indel_max / sm_max if sm_max > 0 else 0.0
    transweight *= indelrate
    indel_vec = indelrate * indel_vec / indel_max if indel_max > 0 else indel_vec
    sm_scaled = sm / sm_max if sm_max > 0 else sm
    stateindel = indel_vec * otto

    n_tok = n_states * n_states
    indels = np.zeros(n_tok, dtype=np.float64)
    for a in range(n_states):
        for b in range(n_states):
            tok = a * n_states + b
            indels[tok] = stateindel[a]
            if a != b:
                if transindel == "constant":
                    raw = 1.0
                elif transindel == "prob":
                    raw = 1.0 - tr[a, b]
                else:
                    raw = sm_scaled[a, b]
                indels[tok] += transweight * raw

    newsm = np.zeros((n_tok, n_tok), dtype=np.float64)
    for i in range(n_tok - 1):
        a_i = i // n_states
        for j in range(i + 1, n_tok):
            a_j = j // n_states
            cost = otto * sm_scaled[a_i, a_j] + (
                indels[i] + indels[j] - stateindel[a_i] - stateindel[a_j]
            )
            newsm[i, j] = cost
            newsm[j, i] = cost
    return indels, newsm


def omstran_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    otto: float,
    indel: float | np.ndarray = 1.0,
    sm: np.ndarray | str = "constant",
    sub_cost: float = 2.0,
    transindel: str = "constant",
    transition_rates: np.ndarray | None = None,
    n_states: int | None = None,
    normalize: bool | str = False,
) -> float:
    """
    Compute TraMineR's transition-sensitive OM distance (``seqdist(method="OMstran")``).

    Args:
        seq_a: First sequence (integer-encoded numpy array).
        seq_b: Second sequence (integer-encoded numpy array).
        otto: Weight of the origin state against the transition, in
            ``[0, 1]``. Required, as in TraMineR.
        indel: State indel cost, scalar or one value per state.
        sm: Substitution cost matrix, or ``"constant"``.
        sub_cost: Constant substitution cost when ``sm="constant"``.
        transindel: How a transition's own indel part is priced:
            ``"constant"`` (1), ``"prob"`` (``1 - transition rate``, needs
            ``transition_rates``), or ``"subcost"`` (the scaled state
            substitution cost).
        transition_rates: State-by-state transition-rate matrix for
            ``transindel="prob"``.
        n_states: Alphabet size. Defaults to the largest state index in
            the pair plus one, or the size of ``sm``. Pass the pool's size
            so tokens and costs are consistent across pairs.
        normalize: ``False``, ``True`` / ``"auto"`` (``"yujianbo"`` for
            OMstran), or an explicit TraMineR norm name.

    Returns:
        OMstran distance.
    """
    if not 0.0 <= otto <= 1.0:
        raise ValueError("'otto' must be a number in [0, 1]")
    m_len, n_len = len(seq_a), len(seq_b)
    if m_len == 0 or n_len == 0:
        raise ValueError("OMstran is not defined for empty sequences.")

    if n_states is None:
        n_states = max(int(seq_a.max()), int(seq_b.max())) + 1
        if not isinstance(sm, str):
            n_states = max(n_states, np.asarray(sm).shape[0])
    if isinstance(sm, str):
        if sm != "constant":
            raise ValueError(f"Unknown substitution method: {sm}")
        sm_matrix = np.full((n_states, n_states), sub_cost, dtype=np.float64)
        np.fill_diagonal(sm_matrix, 0.0)
    else:
        sm_matrix = np.asarray(sm, dtype=np.float64)
        if sm_matrix.shape != (n_states, n_states):
            raise ValueError(
                f"Substitution matrix must be ({n_states}, {n_states}); got {sm_matrix.shape}."
            )

    indels, newsm = omstran_costs(
        n_states, sm_matrix, indel, otto, transindel, transition_rates
    )
    tok_a = _transition_tokens(seq_a, n_states)
    tok_b = _transition_tokens(seq_b, n_states)

    norm = resolve_norm(normalize, auto="yujianbo")
    indel_max = float(indels.max())
    maxscost = max_sub_cost(newsm, indel_max, norm)
    raw = _om_varying_indel_kernel(tok_a, tok_b, indels, newsm)
    maxdist = abs(n_len - m_len) * indel_max + maxscost * min(m_len, n_len)
    return float(
        normalize_distance(raw, maxdist, m_len * indel_max, n_len * indel_max, norm)
    )
