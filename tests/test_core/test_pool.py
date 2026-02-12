"""Tests for SequencePool class."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool


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
        """Test getting encoded sequence."""
        encoded = sequence_pool.get_encoded_sequence(1)

        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 4

    def test_sequence_lengths(self, sequence_pool: SequencePool) -> None:
        """Test getting sequence lengths."""
        lengths = sequence_pool.sequence_lengths()

        assert len(lengths) == 3
        assert lengths["length"].to_list() == [4, 4, 4]

    def test_to_wide_format(self, sequence_pool: SequencePool) -> None:
        """Test converting to wide format."""
        wide = sequence_pool.to_wide_format()

        assert len(wide) == 3
        assert set(wide.columns) == {"id", "0", "1", "2", "3"}

    def test_to_long_format(self, sequence_pool: SequencePool) -> None:
        """Test getting long format."""
        long_df = sequence_pool.to_long_format()

        assert len(long_df) == 12
        assert "id" in long_df.columns
        assert "time" in long_df.columns
        assert "state" in long_df.columns

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
        distances = sequence_pool.compute_distances(method="om")

        assert distances.shape == (3, 3)
        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)
        # Should be symmetric
        assert np.allclose(distances, distances.T)

    def test_compute_distances_hamming(self, sequence_pool: SequencePool) -> None:
        """Test computing Hamming distances."""
        distances = sequence_pool.compute_distances(method="hamming")

        assert distances.shape == (3, 3)
        assert np.allclose(np.diag(distances), 0)

    def test_compute_distances_lcs(self, sequence_pool: SequencePool) -> None:
        """Test computing LCS distances."""
        distances = sequence_pool.compute_distances(method="lcs")

        assert distances.shape == (3, 3)
        assert np.allclose(np.diag(distances), 0)

    def test_invalid_method(self, sequence_pool: SequencePool) -> None:
        """Test error on invalid distance method."""
        with pytest.raises(ValueError, match="Unknown method"):
            sequence_pool.compute_distances(method="invalid")


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
