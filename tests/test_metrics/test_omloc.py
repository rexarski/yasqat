"""Tests for TraMineR's OMloc (yasqat.metrics.omloc).

Reference values are derived by hand from TraMineR 2.2-14
(``src/OMVIdistance.cpp`` with ``OMlocIndelCalculator``, ``R/seqdist.R``); no
R run backs them. ``_ref_omloc`` is a line-by-line Python transcription of
the C++ used to cross-check the numba port on random pairs.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics import omloc_distance, optimal_matching_distance


def _ref_omloc(
    a: list[int], b: list[int], sm: np.ndarray, expcost: float, context: float
) -> float:
    indel = sm.max() * expcost + context
    upper = sm[np.triu_indices(sm.shape[0], k=1)]
    maxscost = min(upper.max(), 2 * indel)
    timecost, localcost = expcost * maxscost, context

    def gi(state: int, prev: int, nxt: int) -> float:
        return timecost + localcost * (sm[prev, state] + sm[nxt, state]) / 2

    m, n = len(a), len(b)
    f = [[0.0] * (n + 1) for _ in range(m + 1)]
    for ii in range(1, m + 1):
        f[ii][0] = f[ii - 1][0] + gi(a[ii - 1], b[0], b[0])
    for jj in range(1, n + 1):
        f[0][jj] = f[0][jj - 1] + gi(b[jj - 1], a[0], a[0])
    prev_j = b[0]
    for j in range(1, n + 1):
        js = b[j - 1]
        prev_i = a[0]
        for i in range(1, m + 1):
            i_s = a[i - 1]
            mn = min(
                f[i - 1][j] + gi(i_s, prev_j, js), f[i][j - 1] + gi(js, prev_i, i_s)
            )
            sub = f[i - 1][j - 1] + (0.0 if i_s == js else sm[i_s, js])
            f[i][j] = min(sub, mn)
            prev_i = i_s
        prev_j = js
    return f[m][n]


def _const_sm(k: int, cost: float = 2.0) -> np.ndarray:
    sm = np.full((k, k), cost)
    np.fill_diagonal(sm, 0.0)
    return sm


class TestOMloc:
    def test_defaults_equal_om_with_half_max_sm_indel(self) -> None:
        """expcost=0.5, context=0: timecost = max(sm)/2, localcost = 0 -> plain OM."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.integers(0, 3, size=rng.integers(1, 8)).astype(np.int32)
            b = rng.integers(0, 3, size=rng.integers(1, 8)).astype(np.int32)
            assert omloc_distance(a, b) == pytest.approx(
                optimal_matching_distance(a, b, indel=1.0, sub_cost=2.0)
            )

    def test_hand_derived_context_case(self) -> None:
        """[0] vs [0,1], expcost=0.25, context=0.5.

        indel param = 2*0.25 + 0.5 = 1, maxscost = 2, timecost = 0.5,
        localcost = 0.5. Inserting state 1 between two 0s costs
        0.5 + 0.5 * (2 + 2) / 2 = 1.5; the DP finds nothing cheaper.
        """
        a = np.array([0], dtype=np.int32)
        b = np.array([0, 1], dtype=np.int32)
        assert omloc_distance(a, b, expcost=0.25, context=0.5) == pytest.approx(1.5)

    def test_cheap_insertion_next_to_similar_states(self) -> None:
        """With a 3-state sm where 1 is close to 0, inserting 1 beside 0s is cheaper than inserting 2."""
        sm = np.array([[0.0, 0.5, 2.0], [0.5, 0.0, 2.0], [2.0, 2.0, 0.0]])
        a = np.array([0, 0], dtype=np.int32)
        near = np.array([0, 1, 0], dtype=np.int32)
        far = np.array([0, 2, 0], dtype=np.int32)
        d_near = omloc_distance(a, near, sm=sm, expcost=0.25, context=0.5)
        d_far = omloc_distance(a, far, sm=sm, expcost=0.25, context=0.5)
        # indel param = 2*0.25+0.5 = 1, maxscost = min(2, 2) = 2, timecost = 0.5
        # insert 1 between 0s: 0.5 + 0.5*(0.5+0.5)/2 = 0.75; insert 2: 0.5 + 0.5*2 = 1.5
        assert d_near == pytest.approx(0.75)
        assert d_far == pytest.approx(1.5)

    def test_matches_cpp_transcription(self) -> None:
        rng = np.random.default_rng(11)
        sm = np.array(
            [
                [0.0, 1.0, 2.0, 1.5],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.5, 2.0, 1.0, 0.0],
            ]
        )
        for _ in range(30):
            a = rng.integers(0, 4, size=rng.integers(1, 9)).astype(np.int32)
            b = rng.integers(0, 4, size=rng.integers(1, 9)).astype(np.int32)
            got = omloc_distance(a, b, sm=sm, expcost=0.3, context=0.2)
            assert got == pytest.approx(
                _ref_omloc(a.tolist(), b.tolist(), sm, 0.3, 0.2)
            )

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.integers(0, 3, size=7).astype(np.int32)
        b = rng.integers(0, 3, size=5).astype(np.int32)
        assert omloc_distance(a, b, expcost=0.2, context=0.4) == pytest.approx(
            omloc_distance(b, a, expcost=0.2, context=0.4)
        )

    def test_identical_is_zero(self) -> None:
        a = np.array([0, 1, 1, 2], dtype=np.int32)
        assert omloc_distance(a, a) == 0.0

    def test_yujianbo_normalization(self) -> None:
        """[0,1,2] vs [0,1,2,3]: raw 1, indel 1, maxscost 2*indel, maxdist = 1 + 2*3 = 7 -> 2/8."""
        a = np.array([0, 1, 2], dtype=np.int32)
        b = np.array([0, 1, 2, 3], dtype=np.int32)
        assert omloc_distance(a, b, normalize=True) == pytest.approx(0.25)
        assert omloc_distance(a, b, normalize="yujianbo") == pytest.approx(0.25)
        assert omloc_distance(a, b, normalize="maxlength") == pytest.approx(1 / 4)
        assert omloc_distance(a, b, normalize="maxdist") == pytest.approx(
            1 / (1 + 2 * 3)
        )

    def test_validation(self) -> None:
        a = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="empty"):
            omloc_distance(a, np.array([], dtype=np.int32))
        with pytest.raises(ValueError, match="expcost"):
            omloc_distance(a, a, expcost=-0.1)
        with pytest.raises(ValueError, match="context"):
            omloc_distance(a, a, expcost=0.8)
        with pytest.raises(ValueError, match="normalization"):
            omloc_distance(a, a, normalize="bogus")


class TestOMlocDispatch:
    def test_pool_matrix_matches_per_pair(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3],
                "time": [0, 1, 2, 0, 1, 2, 0, 1],
                "state": ["A", "B", "C", "A", "A", "C", "B", "C"],
            }
        )
        pool = SequencePool(df)
        dm = pool.compute_distances(method="omloc", expcost=0.25, context=0.5)
        enc = [pool.get_encoded_sequence(i) for i in pool.sequence_ids]
        for i in range(3):
            for j in range(3):
                expected = (
                    0.0
                    if i == j
                    else omloc_distance(enc[i], enc[j], expcost=0.25, context=0.5)
                )
                assert dm.values[i, j] == pytest.approx(expected)
