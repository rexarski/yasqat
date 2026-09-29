"""Pairwise distance engine: metric registry, per-metric preparation, and the
O(n²) driver behind :meth:`SequencePool.compute_distances`.

A metric is registered here as a :class:`MetricSpec`: the free pairwise
function ``fn(seq_a, seq_b, **kwargs) -> float`` and an optional ``prepare``
hook that derives pool-level inputs once (a substitution matrix sized to the
alphabet, DHD position costs, the alphabet size for OMstran, ...) so the
pool itself knows nothing about a metric's internals.

The driver computes each *distinct* sequence pair once and expands the
result to the full matrix, as TraMineR's ``seqdist`` does; pools with many
repeated trajectories pay only for the distinct ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from yasqat.metrics.base import DistanceMatrix, build_substitution_matrix
from yasqat.metrics.chi2 import chi2_distance
from yasqat.metrics.dhd import build_position_costs, dhd_distance
from yasqat.metrics.dtw import dtw_distance
from yasqat.metrics.euclidean import euclidean_distance
from yasqat.metrics.hamming import hamming_distance
from yasqat.metrics.lcp import lcp_distance
from yasqat.metrics.lcs import lcs_distance
from yasqat.metrics.nms import nms_distance, nmsmst_distance, svrspell_distance
from yasqat.metrics.om_variants import (
    om_boundary_weighted_distance,
    om_spell_scaled_distance,
    om_transition_penalty_distance,
)
from yasqat.metrics.omloc import omloc_distance
from yasqat.metrics.omspell import omspell_distance
from yasqat.metrics.omstran import omstran_distance
from yasqat.metrics.optimal_matching import optimal_matching_distance
from yasqat.metrics.rlcp import rlcp_distance
from yasqat.metrics.softdtw import softdtw_distance
from yasqat.metrics.twed import twed_distance

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool

Prepare = Callable[["SequencePool", dict[str, Any]], dict[str, Any]]

SM_METHODS = ("constant", "trate", "indels", "indelslog", "future")


@dataclass(frozen=True)
class MetricSpec:
    """A registered pairwise metric and its optional pool-level preparation."""

    fn: Callable[..., float]
    prepare: Prepare | None = None


def pool_substitution_matrix(
    pool: SequencePool, method: str = "constant", sub_cost: float = 2.0
) -> np.ndarray:
    """Build a substitution matrix for ``pool`` by TraMineR ``seqcost`` method name.

    ``"constant"`` uses ``sub_cost``; ``"trate"``, ``"future"`` derive from the
    pool's transition rates; ``"indels"`` / ``"indelslog"`` from its state
    frequencies. The matrix is sized to the full alphabet.
    """
    from yasqat.statistics.transition import transition_rate_matrix

    n_states = len(pool.alphabet)
    if method == "constant":
        return build_substitution_matrix(n_states, "constant", cost=sub_cost)
    if method in ("trate", "future"):
        rates = transition_rate_matrix(pool, as_counts=False)
        return build_substitution_matrix(n_states, method, transition_rates=rates)
    if method in ("indels", "indelslog"):
        state_col = pool.config.state_column
        counts = pool.data[state_col].value_counts()
        lookup = dict(
            zip(counts[state_col].to_list(), counts["count"].to_list(), strict=True)
        )
        total = float(pool.data.height)
        freq = np.array([lookup.get(s, 0) / total for s in pool.alphabet.states])
        return build_substitution_matrix(n_states, method, state_frequencies=freq)
    raise ValueError(
        f"Unknown method: {method!r} for the substitution matrix. Choose from {SM_METHODS}."
    )


def _prepare_sm(pool: SequencePool, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a string ``sm`` once for the pool (sized to the full alphabet).

    Building the constant matrix per pair dominated the runtime on short
    sequences (issue 17); building ``"trate"`` per pair would also be wrong,
    since a pair does not know the pool's transition rates.
    """
    sm = kwargs.get("sm", "constant")
    if isinstance(sm, str):
        kwargs = dict(kwargs)
        kwargs["sm"] = pool_substitution_matrix(pool, sm, kwargs.get("sub_cost", 2.0))
    return kwargs


def _prepare_dhd(pool: SequencePool, kwargs: dict[str, Any]) -> dict[str, Any]:
    if "position_costs" not in kwargs:
        kwargs = dict(kwargs)
        # Also validates that every sequence has the same length.
        kwargs["position_costs"] = build_position_costs(pool)
    return kwargs


def _prepare_omstran(pool: SequencePool, kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = _prepare_sm(pool, kwargs)
    if "n_states" not in kwargs:
        # Transition tokens must be encoded over the whole alphabet so every
        # pair shares one token space and one cost matrix.
        kwargs["n_states"] = len(pool.alphabet)
    return kwargs


METRICS: dict[str, MetricSpec] = {
    "om": MetricSpec(optimal_matching_distance, _prepare_sm),
    "hamming": MetricSpec(hamming_distance),
    "lcs": MetricSpec(lcs_distance),
    "lcp": MetricSpec(lcp_distance),
    "rlcp": MetricSpec(rlcp_distance),
    "euclidean": MetricSpec(euclidean_distance),
    "chi2": MetricSpec(chi2_distance),
    "dtw": MetricSpec(dtw_distance),
    "softdtw": MetricSpec(softdtw_distance),
    "twed": MetricSpec(twed_distance),
    "dhd": MetricSpec(dhd_distance, _prepare_dhd),
    "omloc": MetricSpec(omloc_distance, _prepare_sm),
    "omspell": MetricSpec(omspell_distance, _prepare_sm),
    "omstran": MetricSpec(omstran_distance, _prepare_omstran),
    "om_boundary": MetricSpec(om_boundary_weighted_distance, _prepare_sm),
    "om_spellscaled": MetricSpec(om_spell_scaled_distance, _prepare_sm),
    "om_transpenalty": MetricSpec(om_transition_penalty_distance, _prepare_sm),
    "nms": MetricSpec(nms_distance),
    "nmsmst": MetricSpec(nmsmst_distance),
    "svrspell": MetricSpec(svrspell_distance),
}


def compute_distance_matrix(
    pool: SequencePool, method: str = "om", n_jobs: int = 1, **kwargs: Any
) -> DistanceMatrix:
    """Pairwise distance matrix over ``pool`` (see ``SequencePool.compute_distances``)."""
    if method not in METRICS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METRICS)}")
    spec = METRICS[method]
    if spec.prepare is not None:
        kwargs = spec.prepare(pool, kwargs)
    metric_fn = spec.fn

    ids = pool.sequence_ids
    encoded = [pool.get_encoded_sequence(i) for i in ids]

    # Distinct sequences only: identical trajectories share one row/column.
    keys: dict[bytes, int] = {}
    inverse = np.empty(len(ids), dtype=np.int64)
    distinct: list[np.ndarray] = []
    for k, seq in enumerate(encoded):
        key = seq.tobytes() + bytes([len(seq) % 256])
        idx = keys.get(key)
        if idx is None:
            idx = len(distinct)
            keys[key] = idx
            distinct.append(seq)
        inverse[k] = idx

    u = len(distinct)
    dist_u = np.zeros((u, u), dtype=np.float64)

    if n_jobs == 1:
        for i in range(u):
            for j in range(i + 1, u):
                d = metric_fn(distinct[i], distinct[j], **kwargs)
                dist_u[i, j] = d
                dist_u[j, i] = d
    else:
        import os
        from concurrent.futures import ThreadPoolExecutor

        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        if workers < 1:
            raise ValueError(f"n_jobs must be >= 1 or -1, got {n_jobs}")
        pairs = [(i, j) for i in range(u) for j in range(i + 1, u)]
        # One task per worker, not one per pair: a Future per pair costs
        # more than a short kernel call and made n_jobs>1 several times
        # slower than sequential on typical 24-step sequences. Each task
        # writes its own disjoint cells, so no locking is needed.
        chunks = [pairs[k::workers] for k in range(workers)]

        def _compute_chunk(chunk: list[tuple[int, int]]) -> None:
            for i, j in chunk:
                d = metric_fn(distinct[i], distinct[j], **kwargs)
                dist_u[i, j] = d
                dist_u[j, i] = d

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # list() re-raises any worker exception on the caller.
            list(executor.map(_compute_chunk, chunks))

    distances = dist_u[np.ix_(inverse, inverse)]
    return DistanceMatrix(values=distances, labels=ids)
