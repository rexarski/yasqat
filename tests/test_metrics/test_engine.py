"""Tests for the pairwise distance engine (yasqat.metrics.engine)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics import (
    METRICS,
    build_substitution_matrix,
    compute_distance_matrix,
    optimal_matching_distance,
    pool_substitution_matrix,
)
from yasqat.statistics import substitution_cost_matrix, transition_rate_matrix


@pytest.fixture
def dup_pool() -> SequencePool:
    """Five sequences, only three distinct (ids 1 == 3, 2 == 5)."""
    seqs = {1: "AAB", 2: "ABC", 3: "AAB", 4: "CCA", 5: "ABC"}
    rows = [(sid, t, st) for sid, seq in seqs.items() for t, st in enumerate(seq)]
    return SequencePool(
        pl.DataFrame(rows, schema=["id", "time", "state"], orient="row")
    )


def _brute_force(pool: SequencePool, **kwargs: object) -> np.ndarray:
    ids = pool.sequence_ids
    enc = [pool.get_encoded_sequence(i) for i in ids]
    out = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j:
                out[i, j] = optimal_matching_distance(enc[i], enc[j], **kwargs)  # type: ignore[arg-type]
    return out


class TestDistinctSequenceDedup:
    def test_matrix_equals_brute_force(self, dup_pool: SequencePool) -> None:
        sm = build_substitution_matrix(3, "constant")
        dm = dup_pool.compute_distances(method="om", sm=sm)
        np.testing.assert_allclose(dm.values, _brute_force(dup_pool, sm=sm))
        # Duplicates sit at distance 0 from each other and share rows.
        assert dm.get_distance(1, 3) == 0.0
        np.testing.assert_allclose(dm.values[0], dm.values[2])
        np.testing.assert_allclose(dm.values[1], dm.values[4])

    def test_threaded_path_equals_brute_force(self, dup_pool: SequencePool) -> None:
        dm = dup_pool.compute_distances(method="om", n_jobs=3)
        np.testing.assert_allclose(
            dm.values,
            _brute_force(dup_pool, sm=build_substitution_matrix(3, "constant")),
        )

    def test_metric_called_once_per_distinct_pair(
        self, dup_pool: SequencePool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, ...]] = []

        def counting(a: np.ndarray, b: np.ndarray, **kw: object) -> float:
            calls.append((len(a), len(b)))
            return float(np.sum(a != b))

        monkeypatch.setitem(METRICS, "counting", type(METRICS["hamming"])(counting))
        dup_pool.compute_distances(method="counting")
        assert len(calls) == 3  # C(3, 2) distinct pairs, not C(5, 2) = 10

    def test_prefix_sequences_are_distinct(self) -> None:
        """AB and ABB must not collide even though one's bytes prefix the other's."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 2],
                "time": [0, 1, 0, 1, 2],
                "state": ["A", "B", "A", "B", "B"],
            }
        )
        dm = SequencePool(df).compute_distances(method="lcs")
        assert dm.values[0, 1] > 0.0


class TestSubstitutionMatrixByName:
    """Regression tests for issue 01: sm reaches the kernel through every input form."""

    @pytest.mark.parametrize("form", ["ndarray", "list", "int", "float32"])
    def test_matrix_forms_agree(self, sequence_pool: SequencePool, form: str) -> None:
        sm = substitution_cost_matrix(sequence_pool, "trate")
        variants = {
            "ndarray": sm,
            "list": sm.tolist(),
            "int": np.rint(sm).astype(int),
            "float32": sm.astype(np.float32),
        }
        dm = sequence_pool.compute_distances(method="om", sm=variants[form])
        expected = _brute_force(
            sequence_pool, sm=np.asarray(variants[form], dtype=float)
        )
        np.testing.assert_allclose(dm.values, expected)

    def test_trate_by_name_equals_matrix(self, sequence_pool: SequencePool) -> None:
        by_name = sequence_pool.compute_distances(method="om", sm="trate")
        by_matrix = sequence_pool.compute_distances(
            method="om", sm=substitution_cost_matrix(sequence_pool, "trate")
        )
        np.testing.assert_allclose(by_name.values, by_matrix.values)
        assert not np.allclose(
            by_name.values, sequence_pool.compute_distances(method="om").values
        )

    @pytest.mark.parametrize(
        "method",
        ["omloc", "omspell", "om_boundary", "om_spellscaled", "om_transpenalty"],
    )
    def test_other_sm_metrics_accept_names(
        self, sequence_pool: SequencePool, method: str
    ) -> None:
        dm = sequence_pool.compute_distances(method=method, sm="indels")
        assert np.isfinite(dm.values).all()
        assert dm.values[0, 1] > 0.0

    def test_omstran_accepts_name(self, sequence_pool: SequencePool) -> None:
        dm = sequence_pool.compute_distances(method="omstran", otto=0.5, sm="trate")
        assert np.isfinite(dm.values).all()

    def test_unknown_name_rejected(self, sequence_pool: SequencePool) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            sequence_pool.compute_distances(method="om", sm="bogus")

    def test_pool_substitution_matrix_methods(
        self, sequence_pool: SequencePool
    ) -> None:
        n = len(sequence_pool.alphabet)
        rates = transition_rate_matrix(sequence_pool)
        np.testing.assert_allclose(
            pool_substitution_matrix(sequence_pool, "future"),
            build_substitution_matrix(n, "future", transition_rates=rates),
        )
        # Frequencies over the fixture: A 3, B 5, C 3, D 1 of 12.
        freq = np.array([3, 5, 3, 1]) / 12
        np.testing.assert_allclose(
            pool_substitution_matrix(sequence_pool, "indelslog"),
            build_substitution_matrix(n, "indelslog", state_frequencies=freq),
        )
        np.testing.assert_allclose(
            pool_substitution_matrix(sequence_pool, "constant", sub_cost=3.0)[0, 1], 3.0
        )


class TestEngineApi:
    def test_unknown_method(self, sequence_pool: SequencePool) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compute_distance_matrix(sequence_pool, "nope")

    def test_free_function_matches_pool_method(
        self, sequence_pool: SequencePool
    ) -> None:
        a = compute_distance_matrix(sequence_pool, "hamming")
        b = sequence_pool.compute_distances(method="hamming")
        np.testing.assert_allclose(a.values, b.values)
        assert a.labels == b.labels

    def test_every_registered_metric_has_a_spec(self) -> None:
        for name, spec in METRICS.items():
            assert callable(spec.fn), name
