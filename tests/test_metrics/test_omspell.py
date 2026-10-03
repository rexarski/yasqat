"""Tests for TraMineR's OMspell (yasqat.metrics.omspell).

Reference values are derived by hand from TraMineR 2.2-14
(``src/OMPerdistance.cpp``, ``R/seqdist.R``); no R run backs them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics import omspell_distance


def _spells(seq: list[int]) -> tuple[list[int], list[float]]:
    st: list[int] = []
    du: list[float] = []
    for x in seq:
        if st and st[-1] == x:
            du[-1] += 1
        else:
            st.append(x)
            du.append(1.0)
    return st, du


def _ref_omspell(
    a: list[int],
    b: list[int],
    sm: np.ndarray,
    indel: float,
    expcost: float,
    tpow: float,
) -> float:
    sa, da = _spells(a)
    sb, db = _spells(b)
    da = [d**tpow - 1 for d in da]
    db = [d**tpow - 1 for d in db]
    m, n = len(sa), len(sb)
    f = [[0.0] * (n + 1) for _ in range(m + 1)]
    for ii in range(1, m + 1):
        f[ii][0] = f[ii - 1][0] + indel + expcost * da[ii - 1]
    for jj in range(1, n + 1):
        f[0][jj] = f[0][jj - 1] + indel + expcost * db[jj - 1]
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            mn = min(
                f[i][j - 1] + indel + expcost * db[j - 1],
                f[i - 1][j] + indel + expcost * da[i - 1],
            )
            if sa[i - 1] == sb[j - 1]:
                sub = f[i - 1][j - 1] + expcost * abs(da[i - 1] - db[j - 1])
            else:
                sub = (
                    f[i - 1][j - 1]
                    + sm[sa[i - 1], sb[j - 1]]
                    + (da[i - 1] + db[j - 1]) * expcost
                )
            f[i][j] = min(sub, mn)
    return f[m][n]


class TestOMspell:
    def test_same_state_different_duration(self) -> None:
        """[0,0,0] vs [0]: one spell each, d' = 2 and 0 -> 0.5 * |2 - 0| = 1."""
        a = np.array([0, 0, 0], dtype=np.int32)
        b = np.array([0], dtype=np.int32)
        assert omspell_distance(a, b) == pytest.approx(1.0)

    def test_different_state_pays_sub_plus_durations(self) -> None:
        """[0,0,0] vs [1]: sm 2 + (2 + 0) * 0.5 = 3; delete+insert costs 2 + 1 = 3 too."""
        a = np.array([0, 0, 0], dtype=np.int32)
        b = np.array([1], dtype=np.int32)
        assert omspell_distance(a, b) == pytest.approx(3.0)

    def test_hand_derived_two_spell_case(self) -> None:
        """[0,0,1] vs [0,1,1] -> spells (0,2)(1,1) vs (0,1)(1,2): 0.5 + 0.5 = 1."""
        a = np.array([0, 0, 1], dtype=np.int32)
        b = np.array([0, 1, 1], dtype=np.int32)
        assert omspell_distance(a, b) == pytest.approx(1.0)

    def test_tpow_transforms_durations(self) -> None:
        """[0,0] vs [0] with tpow=2: d' = 4 - 1 = 3 vs 0 -> 1.5."""
        a = np.array([0, 0], dtype=np.int32)
        b = np.array([0], dtype=np.int32)
        assert omspell_distance(a, b, tpow=2.0) == pytest.approx(1.5)

    def test_identical_is_zero(self) -> None:
        a = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        assert omspell_distance(a, a) == 0.0

    def test_matches_cpp_transcription(self) -> None:
        rng = np.random.default_rng(7)
        sm = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        for _ in range(30):
            a = rng.integers(0, 3, size=rng.integers(1, 10)).astype(np.int32)
            b = rng.integers(0, 3, size=rng.integers(1, 10)).astype(np.int32)
            got = omspell_distance(a, b, sm=sm, indel=0.75, expcost=0.4, tpow=1.5)
            assert got == pytest.approx(
                _ref_omspell(a.tolist(), b.tolist(), sm, 0.75, 0.4, 1.5)
            )

    def test_symmetric(self) -> None:
        a = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
        b = np.array([0, 1, 2, 2], dtype=np.int32)
        assert omspell_distance(a, b) == pytest.approx(omspell_distance(b, a))

    def test_per_state_indel(self) -> None:
        """[0] vs [1] with indels (1, 3): cheapest is sub 2 + 0 (d' = 0 both)."""
        a = np.array([0], dtype=np.int32)
        b = np.array([1], dtype=np.int32)
        assert omspell_distance(a, b, indel=np.array([1.0, 3.0])) == pytest.approx(2.0)
        with pytest.raises(ValueError, match="one value per state"):
            omspell_distance(a, b, indel=np.array([1.0, 2.0, 3.0]))

    def test_normalization_uses_time_point_lengths(self) -> None:
        """[0,0,0] vs [0]: raw 1; maxlength divides by 3 * indel, not by the spell count."""
        a = np.array([0, 0, 0], dtype=np.int32)
        b = np.array([0], dtype=np.int32)
        assert omspell_distance(a, b, normalize="maxlength") == pytest.approx(1 / 3)
        # yujianbo: maxscost = 2*indel = 2, maxdist = |1-3|*1 + 2*1 = 4 -> 2*1/(1+4)
        assert omspell_distance(a, b, normalize=True) == pytest.approx(0.4)

    def test_empty(self) -> None:
        e = np.array([], dtype=np.int32)
        assert omspell_distance(e, e) == 0.0
        assert omspell_distance(np.array([0, 0], dtype=np.int32), e) == pytest.approx(
            1.0 + 0.5
        )


class TestOMspellDispatch:
    def test_pool_matrix_matches_per_pair(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "state": ["A", "A", "C", "A", "B", "C", "B", "B", "B"],
            }
        )
        pool = SequencePool(df)
        dm = pool.compute_distances(method="omspell", expcost=0.5)
        enc = [pool.get_encoded_sequence(i) for i in pool.sequence_ids]
        for i in range(3):
            for j in range(3):
                expected = (
                    0.0 if i == j else omspell_distance(enc[i], enc[j], expcost=0.5)
                )
                assert dm.values[i, j] == pytest.approx(expected)
