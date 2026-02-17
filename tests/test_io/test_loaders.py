"""Tests for data loaders."""

import tempfile

import polars as pl
import pytest

from yasqat.core.sequence import EventSequence, IntervalSequence, StateSequence
from yasqat.io.loaders import (
    infer_sequence_type,
    load_csv,
    load_json,
    load_parquet,
    load_wide_format,
    save_csv,
    save_json,
    save_parquet,
    to_wide_format,
)


@pytest.fixture
def state_sequence_data() -> pl.DataFrame:
    """Create state sequence data for testing."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "state": ["A", "B", "C", "A", "A", "B"],
        }
    )


@pytest.fixture
def event_sequence_data() -> pl.DataFrame:
    """Create event sequence data for testing."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 2],
            "time": [0, 5, 2, 8, 15],
            "state": ["login", "purchase", "login", "view", "logout"],
        }
    )


@pytest.fixture
def interval_sequence_data() -> pl.DataFrame:
    """Create interval sequence data for testing."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "start": [0, 5, 0, 3],
            "end": [5, 10, 3, 8],
            "state": ["working", "meeting", "working", "break"],
        }
    )


class TestCSVLoader:
    """Tests for CSV loading and saving."""

    def test_load_state_sequence_csv(self, state_sequence_data: pl.DataFrame) -> None:
        """Test loading state sequence from CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            state_sequence_data.write_csv(f.name)

            seq = load_csv(f.name, sequence_type="state")

            assert isinstance(seq, StateSequence)
            assert seq.n_sequences() == 2
            assert len(seq.alphabet) == 3

    def test_load_event_sequence_csv(self, event_sequence_data: pl.DataFrame) -> None:
        """Test loading event sequence from CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            event_sequence_data.write_csv(f.name)

            seq = load_csv(f.name, sequence_type="event")

            assert isinstance(seq, EventSequence)
            assert seq.n_sequences() == 2

    def test_load_interval_sequence_csv(
        self, interval_sequence_data: pl.DataFrame
    ) -> None:
        """Test loading interval sequence from CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            interval_sequence_data.write_csv(f.name)

            seq = load_csv(f.name, sequence_type="interval")

            assert isinstance(seq, IntervalSequence)
            assert seq.n_sequences() == 2

    def test_save_and_load_csv(self, state_sequence_data: pl.DataFrame) -> None:
        """Test round-trip save and load."""
        seq = StateSequence(state_sequence_data)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            save_csv(seq, f.name)
            loaded = load_csv(f.name, sequence_type="state")

            assert loaded.n_sequences() == seq.n_sequences()
            assert len(loaded.data) == len(seq.data)

    def test_invalid_sequence_type(self, state_sequence_data: pl.DataFrame) -> None:
        """Test error with invalid sequence type."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            state_sequence_data.write_csv(f.name)

            with pytest.raises(ValueError, match="Unknown sequence_type"):
                load_csv(f.name, sequence_type="invalid")


class TestJSONLoader:
    """Tests for JSON loading and saving."""

    def test_load_state_sequence_json(self, state_sequence_data: pl.DataFrame) -> None:
        """Test loading state sequence from JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            state_sequence_data.write_json(f.name)

            seq = load_json(f.name, sequence_type="state")

            assert isinstance(seq, StateSequence)
            assert seq.n_sequences() == 2

    def test_save_and_load_json(self, state_sequence_data: pl.DataFrame) -> None:
        """Test round-trip save and load."""
        seq = StateSequence(state_sequence_data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            save_json(seq, f.name)
            loaded = load_json(f.name, sequence_type="state")

            assert loaded.n_sequences() == seq.n_sequences()


class TestParquetLoader:
    """Tests for Parquet loading and saving."""

    def test_load_state_sequence_parquet(
        self, state_sequence_data: pl.DataFrame
    ) -> None:
        """Test loading state sequence from Parquet."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            state_sequence_data.write_parquet(f.name)

            seq = load_parquet(f.name, sequence_type="state")

            assert isinstance(seq, StateSequence)
            assert seq.n_sequences() == 2

    def test_save_and_load_parquet(self, state_sequence_data: pl.DataFrame) -> None:
        """Test round-trip save and load."""
        seq = StateSequence(state_sequence_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            save_parquet(seq, f.name)
            loaded = load_parquet(f.name, sequence_type="state")

            assert loaded.n_sequences() == seq.n_sequences()

    def test_parquet_compression(self, state_sequence_data: pl.DataFrame) -> None:
        """Test Parquet with different compression."""
        seq = StateSequence(state_sequence_data)

        for compression in ["zstd", "snappy"]:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                save_parquet(seq, f.name, compression=compression)
                loaded = load_parquet(f.name, sequence_type="state")

                assert loaded.n_sequences() == seq.n_sequences()


class TestInferSequenceType:
    """Tests for sequence type inference."""

    def test_infer_state_sequence(self, state_sequence_data: pl.DataFrame) -> None:
        """Test inferring state sequence type."""
        seq_type = infer_sequence_type(state_sequence_data)

        # Has id, time, state -> state sequence
        assert seq_type == "state"

    def test_infer_interval_sequence(
        self, interval_sequence_data: pl.DataFrame
    ) -> None:
        """Test inferring interval sequence type."""
        seq_type = infer_sequence_type(interval_sequence_data)

        # Has start and end -> interval sequence
        assert seq_type == "interval"


class TestWideFormat:
    """Tests for wide format conversion."""

    def test_to_wide_format(self, state_sequence_data: pl.DataFrame) -> None:
        """Test converting to wide format."""
        seq = StateSequence(state_sequence_data)

        wide = to_wide_format(seq)

        # Should have id column plus time columns
        assert "id" in wide.columns
        assert len(wide) == 2  # 2 sequences

    def test_load_wide_format(self) -> None:
        """Test reading wide format data."""
        # Create wide format data
        wide_data = pl.DataFrame(
            {
                "id": [1, 2],
                "0": ["A", "A"],
                "1": ["B", "A"],
                "2": ["C", "B"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            wide_data.write_csv(f.name)

            long_data = load_wide_format(f.name)

            assert "id" in long_data.columns
            assert "time" in long_data.columns
            assert "state" in long_data.columns
            # 2 sequences * 3 time points = 6 rows
            assert len(long_data) == 6

    def test_round_trip_wide_long(self, state_sequence_data: pl.DataFrame) -> None:
        """Test round-trip wide to long conversion."""
        seq = StateSequence(state_sequence_data)

        # To wide
        wide = to_wide_format(seq)

        # Save and reload as wide
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            wide.write_csv(f.name)

            long_data = load_wide_format(f.name)

            # Should have same number of records
            assert len(long_data) == len(state_sequence_data)
