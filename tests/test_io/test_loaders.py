"""Tests for data loaders."""

import tempfile

import polars as pl
import pytest

from yasqat.core.sequence import (
    EventSequence,
    IntervalSequence,
    SequenceConfig,
    StateSequence,
)
from yasqat.io.loaders import (
    infer_sequence_type,
    load_csv,
    load_dataframe,
    load_json,
    load_parquet,
    save_csv,
    save_json,
    save_parquet,
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


class TestInferSequenceTypeSimplified:
    def test_interval_detection(self) -> None:
        """Should detect interval when start and end columns present."""
        df = pl.DataFrame({"id": [1], "start": [0], "end": [5], "state": ["A"]})
        assert infer_sequence_type(df) == "interval"

    def test_state_detection(self) -> None:
        """Should return 'state' when time column present (no start/end)."""
        df = pl.DataFrame({"id": [1, 1], "time": [0, 1], "state": ["A", "B"]})
        assert infer_sequence_type(df) == "state"

    def test_default_state(self) -> None:
        """Should default to 'state' when no time columns found."""
        df = pl.DataFrame({"id": [1], "state": ["A"]})
        assert infer_sequence_type(df) == "state"


class TestLoadDataFrame:
    """Tests for load_dataframe function."""

    def test_load_basic_dataframe(self) -> None:
        """Test loading a polars DataFrame into a SequencePool."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "state": ["A", "B", "C", "A", "A", "B"],
            }
        )
        pool = load_dataframe(df)
        assert len(pool) == 2
        assert pool[1] == ["A", "B", "C"]

    def test_load_with_custom_config(self) -> None:
        """Test loading with custom column names."""
        df = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "ts": [0, 1, 0, 1],
                "event": ["X", "Y", "X", "X"],
            }
        )
        config = SequenceConfig(
            id_column="user_id",
            time_column="ts",
            state_column="event",
        )
        pool = load_dataframe(df, config=config)
        assert len(pool) == 2
        assert pool[1] == ["X", "Y"]

    def test_load_with_drop_nulls(self) -> None:
        """Test that null states are dropped."""
        df = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 1, 2, 0, 1],
                "state": ["A", None, "B", "C", "C"],
            }
        )
        pool = load_dataframe(df, drop_nulls=True)
        assert pool[1] == ["A", "B"]
        assert pool[2] == ["C", "C"]

    def test_load_missing_column_raises(self) -> None:
        """Test that missing columns raise ValueError."""
        df = pl.DataFrame({"id": [1], "time": [0]})
        with pytest.raises(ValueError, match="Missing required column"):
            load_dataframe(df)


class TestLoadDataframeAlphabetValidation:
    def test_mismatched_alphabet_raises(self) -> None:
        """load_dataframe should raise ValueError if data has states not in alphabet."""
        from yasqat.core.alphabet import Alphabet
        from yasqat.io import load_dataframe

        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "A", "C"],
            }
        )
        # Alphabet only has A and B, but data has C
        alphabet = Alphabet(states=("A", "B"))
        with pytest.raises(ValueError, match=r"not in.*alphabet"):
            load_dataframe(df, alphabet=alphabet)

    def test_matching_alphabet_works(self) -> None:
        """load_dataframe should work when alphabet covers all data states."""
        from yasqat.core.alphabet import Alphabet
        from yasqat.io import load_dataframe

        df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "A", "B"],
            }
        )
        alphabet = Alphabet(states=("A", "B", "C"))  # superset is fine
        pool = load_dataframe(df, alphabet=alphabet)
        assert len(pool) == 2


class TestIOAutoImport:
    def test_import_yasqat_io(self) -> None:
        """Importing yasqat should make yasqat.io available."""
        import yasqat

        assert hasattr(yasqat, "io")
        assert hasattr(yasqat.io, "load_csv")
        assert hasattr(yasqat.io, "load_dataframe")

    def test_yasqat_io_does_not_shadow_stdlib(self) -> None:
        """yasqat.io should not shadow the stdlib io module."""
        import io as stdlib_io

        import yasqat

        assert stdlib_io is not yasqat.io
        assert hasattr(stdlib_io, "StringIO")
