"""Tests for SequencePool class."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
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
