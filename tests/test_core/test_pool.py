"""Tests for SequencePool class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.core.sequence import StateSequence
from yasqat.metrics.base import DistanceMatrix


class TestSequencePool:
    """Tests for SequencePool class."""

    def test_create_pool(self, simple_sequence_data: pl.DataFrame) -> None:
        """Test creating a sequence pool."""
        pool = SequencePool(simple_sequence_data)

        assert len(pool) == 3
        assert pool.sequence_ids == [1, 2, 3]

    def test_get_sequence(self, sequence_pool: SequencePool) -> None:
        """Test getting a sequence from the pool."""
        seq = sequence_pool.get_sequence(1)

        assert seq == ["A", "A", "B", "C"]

    def test_getitem(self, sequence_pool: SequencePool) -> None:
        """Test getting sequence using bracket notation."""
        seq = sequence_pool[2]

        assert seq == ["A", "B", "B", "C"]

    def test_get_encoded_sequence(self, sequence_pool: SequencePool) -> None:
        """Test getting encoded sequence and roundtrip decode."""
        encoded = sequence_pool.get_encoded_sequence(1)

        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 4
        # Verify actual encoded values match alphabet ordering
        # Alphabet is sorted: A=0, B=1, C=2, D=3
        # Sequence 1 is [A, A, B, C] -> [0, 0, 1, 2]
        assert encoded.tolist() == [0, 0, 1, 2]
        # Roundtrip: decode should recover the original states
        decoded = sequence_pool.alphabet.decode(encoded)
        assert decoded == sequence_pool.get_sequence(1)

    def test_sequence_lengths(self, sequence_pool: SequencePool) -> None:
        """Test getting sequence lengths."""
        lengths = sequence_pool.sequence_lengths()

        assert len(lengths) == 3
        assert lengths["length"].to_list() == [4, 4, 4]

    def test_filter_by_length(self, large_sequence_data: pl.DataFrame) -> None:
        """Test filtering sequences by length."""
        pool = SequencePool(large_sequence_data)

        # All sequences have length 20
        filtered = pool.filter_by_length(min_length=10, max_length=30)
        assert len(filtered) == len(pool)

        # No sequences should pass this filter
        filtered = pool.filter_by_length(min_length=100)
        assert len(filtered) == 0

    def test_sample(self, sequence_pool: SequencePool) -> None:
        """Test sampling sequences."""
        sampled = sequence_pool.sample(2, seed=42)

        assert len(sampled) == 2

    def test_describe(self, sequence_pool: SequencePool) -> None:
        """Test getting pool description."""
        desc = sequence_pool.describe()

        assert desc["n_sequences"] == 3
        assert desc["n_states"] == 4
        assert desc["min_length"] == 4
        assert desc["max_length"] == 4
        assert desc["mean_length"] == 4.0

    def test_iteration(self, sequence_pool: SequencePool) -> None:
        """Test iterating over pool."""
        seq_ids = list(sequence_pool)

        assert seq_ids == [1, 2, 3]

    def test_compute_distances_om(self, sequence_pool: SequencePool) -> None:
        """Test computing OM distances."""
        dm = sequence_pool.compute_distances(method="om")

        assert isinstance(dm, DistanceMatrix)
        assert dm.values.shape == (3, 3)
        assert dm.labels == [1, 2, 3]
        # Diagonal should be zero
        assert np.allclose(np.diag(dm.values), 0)
        # Should be symmetric
        assert np.allclose(dm.values, dm.values.T)

    def test_compute_distances_parallel(self, sequence_pool: SequencePool) -> None:
        """Test parallel distance computation matches sequential."""
        dm_seq = sequence_pool.compute_distances(method="lcs", n_jobs=1)
        dm_par = sequence_pool.compute_distances(method="lcs", n_jobs=2)
        assert np.allclose(dm_seq.values, dm_par.values)

    def test_compute_distances_hamming(self, sequence_pool: SequencePool) -> None:
        """Test computing Hamming distances."""
        dm = sequence_pool.compute_distances(method="hamming")

        assert isinstance(dm, DistanceMatrix)
        assert dm.values.shape == (3, 3)
        assert np.allclose(np.diag(dm.values), 0)

    def test_compute_distances_lcs(self, sequence_pool: SequencePool) -> None:
        """Test computing LCS distances."""
        dm = sequence_pool.compute_distances(method="lcs")

        assert isinstance(dm, DistanceMatrix)
        assert dm.values.shape == (3, 3)
        assert np.allclose(np.diag(dm.values), 0)

    def test_invalid_method(self, sequence_pool: SequencePool) -> None:
        """Test error on invalid distance method."""
        with pytest.raises(ValueError, match="Unknown method"):
            sequence_pool.compute_distances(method="invalid")

    def test_compute_distances_dhd_auto_position_costs(self) -> None:
        """DHD builds position costs from the pool when not supplied.

        Pool: [A,A], [A,B], [B,B].  Frequencies: t0 A=2/3 B=1/3,
        t1 A=1/3 B=2/3.  cost(A,B,t) = 1 - (fa*fb)/0.25 = 1/9 at both
        positions, so d(1,2)=1/9, d(1,3)=2/9, d(2,3)=1/9.
        """
        pool = SequencePool(
            pl.DataFrame(
                {
                    "id": [1, 1, 2, 2, 3, 3],
                    "time": [0, 1, 0, 1, 0, 1],
                    "state": ["A", "A", "A", "B", "B", "B"],
                }
            )
        )
        dm = pool.compute_distances(method="dhd")

        assert dm.values[0, 1] == pytest.approx(1 / 9)
        assert dm.values[0, 2] == pytest.approx(2 / 9)
        assert dm.values[1, 2] == pytest.approx(1 / 9)
        assert np.allclose(dm.values, dm.values.T)
        assert np.allclose(np.diag(dm.values), 0)

    def test_compute_distances_dhd_explicit_position_costs(self) -> None:
        """An explicitly supplied position_costs array is respected."""
        pool = SequencePool(
            pl.DataFrame(
                {
                    "id": [1, 1, 2, 2],
                    "time": [0, 1, 0, 1],
                    "state": ["A", "A", "B", "B"],
                }
            )
        )
        zero_costs = np.zeros((2, 2, 2), dtype=np.float64)
        dm = pool.compute_distances(method="dhd", position_costs=zero_costs)

        assert np.allclose(dm.values, 0.0)

    def test_compute_distances_dhd_matches_free_function(
        self, sequence_pool: SequencePool
    ) -> None:
        """Pool-level DHD equals the free function with pool-built costs."""
        from yasqat.metrics.dhd import build_position_costs, dhd_distance

        costs = build_position_costs(sequence_pool)
        dm = sequence_pool.compute_distances(method="dhd")

        a = sequence_pool.get_encoded_sequence(1)
        b = sequence_pool.get_encoded_sequence(2)
        assert dm.values[0, 1] == pytest.approx(dhd_distance(a, b, costs))

    def test_compute_distances_dhd_unequal_lengths_raises(self) -> None:
        """DHD on an unequal-length pool raises a clear ValueError."""
        pool = SequencePool(
            pl.DataFrame(
                {
                    "id": [1, 1, 1, 2, 2],
                    "time": [0, 1, 2, 0, 1],
                    "state": ["A", "B", "C", "A", "B"],
                }
            )
        )
        with pytest.raises(ValueError, match="same length"):
            pool.compute_distances(method="dhd")


class TestExtractSequencesPerformance:
    """Tests that _extract_sequences uses efficient group_by approach."""

    def test_sequences_match_original_order(
        self, simple_sequence_data: pl.DataFrame
    ) -> None:
        """Test that extracted sequences preserve time order within each ID."""
        pool = SequencePool(simple_sequence_data)
        assert pool[1] == ["A", "A", "B", "C"]
        assert pool[2] == ["A", "B", "B", "C"]
        assert pool[3] == ["B", "B", "C", "D"]

    def test_describe_handles_nulls_gracefully(
        self, simple_sequence_data: pl.DataFrame
    ) -> None:
        """Test that describe() never returns None for length stats."""
        pool = SequencePool(simple_sequence_data)
        desc = pool.describe()
        assert desc["min_length"] is not None
        assert desc["max_length"] is not None
        assert desc["mean_length"] is not None
        assert desc["median_length"] is not None


class TestPoolEdgeCases:
    """Tests for edge cases in SequencePool."""

    def test_null_state_raises_value_error(self) -> None:
        """A null state is rejected at the boundary, not as a TypeError later."""
        df = pl.DataFrame(
            {"id": [1, 1, 2, 2], "time": [0, 1, 0, 1], "state": ["A", None, "A", "B"]}
        )
        with pytest.raises(ValueError, match="contains 1 null value"):
            SequencePool(df)

    def test_empty_pool_error(self) -> None:
        """Creating a pool from an empty DataFrame should raise an error."""
        empty_df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "time": pl.Series([], dtype=pl.Int64),
                "state": pl.Series([], dtype=pl.Utf8),
            }
        )
        # SequencePool should either raise or produce a pool with 0 sequences
        pool = SequencePool(empty_df)
        assert len(pool) == 0
        assert pool.sequence_ids == []

    def test_single_sequence_pool(self) -> None:
        """A pool with just one sequence should work correctly."""
        df = pl.DataFrame(
            {"id": [1, 1, 1], "time": [0, 1, 2], "state": ["A", "B", "A"]}
        )
        pool = SequencePool(df)
        assert len(pool) == 1
        assert pool.sequence_ids == [1]
        assert pool[1] == ["A", "B", "A"]
        desc = pool.describe()
        assert desc["n_sequences"] == 1
        assert desc["min_length"] == 3
        assert desc["max_length"] == 3


class TestRecodeStates:
    """Tests for SequencePool.recode_states method."""

    def test_rename_state(self, sequence_pool: SequencePool) -> None:
        """Test renaming a single state."""
        recoded = sequence_pool.recode_states({"A": "X"})
        assert "X" in recoded.alphabet.states
        assert "A" not in recoded.alphabet.states
        assert recoded[1] == ["X", "X", "B", "C"]

    def test_merge_states(self, sequence_pool: SequencePool) -> None:
        """Test merging multiple states into one."""
        recoded = sequence_pool.recode_states({"A": "X", "B": "X"})
        assert "X" in recoded.alphabet.states
        assert "A" not in recoded.alphabet.states
        assert "B" not in recoded.alphabet.states
        # Original seq 1: A,A,B,C -> X,X,X,C
        assert recoded[1] == ["X", "X", "X", "C"]

    def test_alphabet_reduced(self, sequence_pool: SequencePool) -> None:
        """Test that alphabet is rebuilt with fewer states after merge."""
        recoded = sequence_pool.recode_states({"A": "X", "B": "X"})
        # Original: A, B, C, D -> Merged: X, C, D
        assert len(recoded.alphabet) == 3

    def test_unmapped_states_preserved(self, sequence_pool: SequencePool) -> None:
        """Test that states not in the mapping are kept as-is."""
        recoded = sequence_pool.recode_states({"A": "X"})
        # Seq 3: B,B,C,D should be unchanged
        assert recoded[3] == ["B", "B", "C", "D"]

    def test_empty_mapping(self, sequence_pool: SequencePool) -> None:
        """Test with empty mapping (no changes)."""
        recoded = sequence_pool.recode_states({})
        assert recoded[1] == sequence_pool[1]
        assert len(recoded.alphabet) == len(sequence_pool.alphabet)

    def test_returns_new_pool(self, sequence_pool: SequencePool) -> None:
        """Test that recode returns a new pool, not modifying the original."""
        recoded = sequence_pool.recode_states({"A": "X"})
        assert sequence_pool[1] == ["A", "A", "B", "C"]
        assert recoded[1] == ["X", "X", "B", "C"]


class TestCoerce:
    """Tests for SequencePool.coerce, the single union-normalization seam."""

    def test_pool_is_returned_unchanged(self, sequence_pool: SequencePool) -> None:
        """A SequencePool coerces to itself (identity, no rebuild)."""
        assert SequencePool.coerce(sequence_pool) is sequence_pool

    def test_state_sequence_is_converted(self, state_sequence: StateSequence) -> None:
        """A StateSequence coerces to an equivalent SequencePool."""
        pool = SequencePool.coerce(state_sequence)

        assert isinstance(pool, SequencePool)
        assert pool.sequence_ids == state_sequence.sequence_ids
        assert pool.alphabet == state_sequence.alphabet
        assert pool.get_sequence(1) == ["A", "A", "B", "C"]


class TestPoolFilter:
    """SequencePool.filter returns a typed pool that chains into analysis.

    Fixture sequences: 1 = A A B C, 2 = A B B C, 3 = B B C D.
    """

    def test_single_criterion_keeps_matching_ids(
        self, sequence_pool: SequencePool
    ) -> None:
        from yasqat.filters import ContainsStateCriterion

        situation = sequence_pool.filter(ContainsStateCriterion(states=["A"]))
        assert isinstance(situation, SequencePool)
        assert situation.sequence_ids == [1, 2]
        assert situation.get_sequence(2) == ["A", "B", "B", "C"]

    def test_and_combination(self, sequence_pool: SequencePool) -> None:
        from yasqat.filters import ContainsStateCriterion, StartsWithCriterion

        situation = sequence_pool.filter(
            [StartsWithCriterion(states=["B"]), ContainsStateCriterion(states=["D"])]
        )
        assert situation.sequence_ids == [3]

    def test_or_combination(self, sequence_pool: SequencePool) -> None:
        from yasqat.filters import ContainsStateCriterion, StartsWithCriterion

        situation = sequence_pool.filter(
            [StartsWithCriterion(states=["B"]), ContainsStateCriterion(states=["A"])],
            combine="or",
        )
        assert situation.sequence_ids == [1, 2, 3]

    def test_alphabet_and_config_are_preserved(
        self, sequence_pool: SequencePool
    ) -> None:
        from yasqat.filters import ContainsStateCriterion

        # "D" only occurs in sequence 3, which is filtered out; the alphabet
        # must still carry it so encodings match the parent pool.
        situation = sequence_pool.filter(ContainsStateCriterion(states=["A"]))
        assert situation.alphabet == sequence_pool.alphabet
        assert situation.config == sequence_pool.config
        assert list(situation.get_encoded_sequence(1)) == list(
            sequence_pool.get_encoded_sequence(1)
        )

    def test_empty_result_is_an_empty_pool(self, sequence_pool: SequencePool) -> None:
        from yasqat.filters import LengthCriterion

        situation = sequence_pool.filter(LengthCriterion(min_length=10))
        assert len(situation) == 0
        assert situation.sequence_ids == []

    def test_chains_into_compute_distances(self, sequence_pool: SequencePool) -> None:
        from yasqat.filters import ContainsStateCriterion

        situation = sequence_pool.filter(ContainsStateCriterion(states=["A"]))
        dm = situation.compute_distances(method="hamming")
        assert dm.values.shape == (2, 2)
        assert dm.labels == [1, 2]
        # A A B C vs A B B C differ at exactly one position.
        assert dm.values[0, 1] == pytest.approx(1.0)


class TestOmDispatchHoistsSubstitutionMatrix:
    """compute_distances builds the constant OM matrix once per call.

    Regression for .scratch/issues/17: the hoisted pool-wide matrix must give
    exactly the per-pair results of ``optimal_matching_distance``.
    """

    @staticmethod
    def _per_pair(pool: SequencePool, **kwargs: object) -> np.ndarray:
        from yasqat.metrics import optimal_matching_distance

        ids = pool.sequence_ids
        enc = [pool.get_encoded_sequence(i) for i in ids]
        n = len(ids)
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                out[i, j] = out[j, i] = optimal_matching_distance(
                    enc[i],
                    enc[j],
                    **kwargs,  # type: ignore[arg-type]
                )
        return out

    def test_default_matches_per_pair(self, sequence_pool: SequencePool) -> None:
        dm = sequence_pool.compute_distances(method="om")
        np.testing.assert_allclose(dm.values, self._per_pair(sequence_pool))

    def test_sub_cost_is_honoured(self, sequence_pool: SequencePool) -> None:
        # With indel=1.0 a substitution only wins when it costs < 2 indels,
        # so use 1.5 to make the hoisted cost observable.
        dm = sequence_pool.compute_distances(method="om", sub_cost=1.5)
        expected = self._per_pair(sequence_pool, sub_cost=1.5)
        np.testing.assert_allclose(dm.values, expected)
        assert dm.values[0, 1] == pytest.approx(1.5)
        default = sequence_pool.compute_distances(method="om").values
        assert default[0, 1] == pytest.approx(2.0)

    def test_explicit_matrix_is_not_overridden(
        self, sequence_pool: SequencePool
    ) -> None:
        n = len(sequence_pool.alphabet.states)
        sm = np.full((n, n), 5.0)
        np.fill_diagonal(sm, 0.0)
        dm = sequence_pool.compute_distances(method="om", sm=sm)
        np.testing.assert_allclose(dm.values, self._per_pair(sequence_pool, sm=sm))

    def test_parallel_matches_sequential(self, sequence_pool: SequencePool) -> None:
        dm_seq = sequence_pool.compute_distances(method="om", n_jobs=1)
        dm_par = sequence_pool.compute_distances(method="om", n_jobs=2)
        np.testing.assert_allclose(dm_seq.values, dm_par.values)

    def test_parallel_uneven_chunks_cover_every_pair(self) -> None:
        """7 sequences = 21 pairs over 4 workers: chunks of 6, 5, 5, 5."""
        rng = np.random.default_rng(0)
        n, t = 7, 6
        df = pl.DataFrame(
            {
                "id": np.repeat(np.arange(n), t),
                "time": np.tile(np.arange(t), n),
                "state": rng.choice(["A", "B", "C"], size=n * t),
            }
        )
        pool = SequencePool(df)
        dm_seq = pool.compute_distances(method="om", n_jobs=1)
        dm_par = pool.compute_distances(method="om", n_jobs=4)
        np.testing.assert_allclose(dm_seq.values, dm_par.values)
        assert (dm_par.values[np.triu_indices(n, k=1)] > 0).all()

    def test_n_jobs_zero_rejected(self, sequence_pool: SequencePool) -> None:
        with pytest.raises(ValueError, match="n_jobs"):
            sequence_pool.compute_distances(method="om", n_jobs=0)
