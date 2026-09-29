"""Tests for TraMineR's OMstran (yasqat.metrics.omstran).

Reference values are derived by hand from TraMineR 2.2-14
``R/seqdist-OMstran.R`` (defaults ``previous=FALSE``, ``add.column=TRUE``,
``transindel="constant"``); no R run backs them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics import omstran_costs, omstran_distance, optimal_matching_distance


class TestOMstranCosts:
    def test_two_state_constant_costs(self) -> None:
        """otto=0.5, indel 1, sm 2: indelrate 0.5, transweight 0.25, stateindel 0.25."""
        sm = np.array([[0.0, 2.0], [2.0, 0.0]])
        indels, newsm = omstran_costs(2, sm, 1.0, otto=0.5)
        # tokens: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
        assert indels.tolist() == pytest.approx([0.25, 0.5, 0.5, 0.25])
        assert newsm[0, 1] == pytest.approx(0.25)  # same origin, one transition part
        assert newsm[0, 3] == pytest.approx(0.5)  # different origin: 0.5 * 1
        assert newsm[1, 2] == pytest.approx(1.0)  # 0.5 + 0.25 + 0.25
        assert newsm[0, 0] == 0.0

    def test_transindel_variants(self) -> None:
        sm = np.array([[0.0, 2.0], [2.0, 0.0]])
        tr = np.array([[0.9, 0.1], [0.4, 0.6]])
        i_prob, _ = omstran_costs(2, sm, 1.0, 0.5, "prob", tr)
        assert i_prob[1] == pytest.approx(0.25 + 0.25 * (1 - 0.1))
        i_sub, _ = omstran_costs(2, sm, 1.0, 0.5, "subcost")
        assert i_sub[1] == pytest.approx(0.25 + 0.25 * 1.0)  # sm scaled to 1
        with pytest.raises(ValueError, match="transition_rates"):
            omstran_costs(2, sm, 1.0, 0.5, "prob")
        with pytest.raises(ValueError, match="transindel"):
            omstran_costs(2, sm, 1.0, 0.5, "other")


class TestOMstran:
    def test_hand_derived_case(self) -> None:
        """[0,0,1] vs [0,0,0], otto=0.5: tokens A,B,C vs A,A,A -> 0 + 0.25 + 0.5 = 0.75."""
        a = np.array([0, 0, 1], dtype=np.int32)
        b = np.array([0, 0, 0], dtype=np.int32)
        assert omstran_distance(a, b, otto=0.5) == pytest.approx(0.75)

    def test_otto_one_is_om_over_states_scaled_by_max_sm(self) -> None:
        """otto=1 drops the transition part: OM on origin states with costs / max(sm)."""
        rng = np.random.default_rng(2)
        for _ in range(20):
            a = rng.integers(0, 3, size=rng.integers(1, 8)).astype(np.int32)
            b = rng.integers(0, 3, size=rng.integers(1, 8)).astype(np.int32)
            got = omstran_distance(a, b, otto=1.0, n_states=3)
            expected = optimal_matching_distance(a, b, indel=1.0, sub_cost=2.0) / 2.0
            assert got == pytest.approx(expected)

    def test_transition_change_costs_more_than_state_change_alone(self) -> None:
        """Same origin states, different next state: only the transition part is paid."""
        a = np.array([0, 1], dtype=np.int32)
        b = np.array([0, 0], dtype=np.int32)
        # tokens (0,1),(1,1) vs (0,0),(0,0): sub 0.25 + sub 0.5 = 0.75; alternatives cost more.
        assert omstran_distance(a, b, otto=0.5) == pytest.approx(0.75)

    def test_symmetric_and_identity(self) -> None:
        a = np.array([0, 1, 1, 2], dtype=np.int32)
        b = np.array([2, 1, 0], dtype=np.int32)
        assert omstran_distance(a, a, otto=0.3) == 0.0
        assert omstran_distance(a, b, otto=0.3) == pytest.approx(
            omstran_distance(b, a, otto=0.3)
        )

    def test_normalization(self) -> None:
        """[0,0,1] vs [0,0,0]: raw 0.75, max indel 0.5, yujianbo maxscost = 1, maxdist = 3."""
        a = np.array([0, 0, 1], dtype=np.int32)
        b = np.array([0, 0, 0], dtype=np.int32)
        assert omstran_distance(a, b, otto=0.5, normalize=True) == pytest.approx(
            2 * 0.75 / (0.75 + 3.0)
        )
        assert omstran_distance(a, b, otto=0.5, normalize="maxlength") == pytest.approx(
            0.75 / (3 * 0.5)
        )

    def test_validation(self) -> None:
        a = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="otto"):
            omstran_distance(a, a, otto=1.5)
        with pytest.raises(ValueError, match="empty"):
            omstran_distance(a, np.array([], dtype=np.int32), otto=0.5)
        with pytest.raises(ValueError, match="must be \\(3, 3\\)"):
            omstran_distance(a, a, otto=0.5, sm=np.zeros((2, 2)), n_states=3)


class TestOMstranDispatch:
    def test_pool_passes_alphabet_size(self) -> None:
        """A pair that never shows state C must still use the 3-state token space."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "state": ["A", "A", "B", "A", "A", "A", "C", "B", "A"],
            }
        )
        pool = SequencePool(df)
        dm = pool.compute_distances(method="omstran", otto=0.5)
        enc = [pool.get_encoded_sequence(i) for i in pool.sequence_ids]
        assert dm.values[0, 1] == pytest.approx(
            omstran_distance(enc[0], enc[1], otto=0.5, n_states=3)
        )
        assert dm.values[0, 1] == pytest.approx(0.75)
        with pytest.raises(TypeError):
            pool.compute_distances(method="omstran")  # otto is required
