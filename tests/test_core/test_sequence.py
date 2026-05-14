"""Tests for StateSequence classes."""

from __future__ import annotations

import polars as pl
import pytest

from yasqat.core.alphabet import Alphabet
from yasqat.core.sequence import (
    SequenceConfig,
    StateSequence,
)


class TestStateSequence:
    """Tests for StateSequence class."""

    def test_create_sequence(self, simple_sequence_data: pl.DataFrame) -> None:
        """Test creating a state sequence with correct data values."""
        seq = StateSequence(simple_sequence_data)

        assert len(seq) == 12
        assert seq.n_sequences() == 3
        assert seq.sequence_ids == [1, 2, 3]
        # Verify actual state values for each sequence
        assert seq.get_states_for_sequence(1) == ["A", "A", "B", "C"]
        assert seq.get_states_for_sequence(2) == ["A", "B", "B", "C"]
        assert seq.get_states_for_sequence(3) == ["B", "B", "C", "D"]

    def test_single_element_sequence(self) -> None:
        """A sequence with one time step should work."""
        data = pl.DataFrame({"id": [1], "time": [0], "state": ["A"]})
        seq = StateSequence(data)
        assert len(seq) == 1
        assert seq.n_sequences() == 1
        assert seq.get_states_for_sequence(1) == ["A"]
        # SPS should have exactly one spell
        sps = seq.to_sps()
        assert len(sps) == 1
        assert sps["duration"].item() == 1

    def test_to_dss_all_same_state(self) -> None:
        """DSS of a constant sequence should have one spell."""
        data = pl.DataFrame(
            {"id": [1, 1, 1, 1], "time": [0, 1, 2, 3], "state": ["A", "A", "A", "A"]}
        )
        seq = StateSequence(data)
        dss = seq.to_dss()
        # All same state -> only one distinct successive state
        assert len(dss) == 1
        assert dss["state"].to_list() == ["A"]

    def test_sequence_with_custom_config(self) -> None:
        """Test creating sequence with custom column names."""
        data = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "timestamp": [0, 1, 0, 1],
                "status": ["A", "B", "C", "D"],
            }
        )

        config = SequenceConfig(
            id_column="user_id",
            time_column="timestamp",
            state_column="status",
        )

        seq = StateSequence(data, config=config)

        assert seq.n_sequences() == 2
        assert len(seq.alphabet) == 4

    def test_missing_column_error(self) -> None:
        """Test error when required column is missing."""
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [0, 1],
                # Missing 'state' column
            }
        )

        with pytest.raises(ValueError, match="Missing required column"):
            StateSequence(data)

    def test_get_sequence(self, state_sequence: StateSequence) -> None:
        """Test getting a specific sequence."""
        seq_data = state_sequence.get_sequence(1)

        assert len(seq_data) == 4
        assert seq_data["state"].to_list() == ["A", "A", "B", "C"]

    def test_get_states_for_sequence(self, state_sequence: StateSequence) -> None:
        """Test getting states list for a sequence."""
        states = state_sequence.get_states_for_sequence(2)

        assert states == ["A", "B", "B", "C"]

    def test_to_sts(self, state_sequence: StateSequence) -> None:
        """Test getting STS format."""
        sts = state_sequence.to_sts()

        assert "id" in sts.columns
        assert "time" in sts.columns
        assert "state" in sts.columns
        assert len(sts) == 12

    def test_to_sps(self, state_sequence: StateSequence) -> None:
        """Test converting to SPS (spell) format."""
        sps = state_sequence.to_sps()

        assert "id" in sps.columns
        assert "spell_id" in sps.columns
        assert "state" in sps.columns
        assert "start" in sps.columns
        assert "end" in sps.columns
        assert "duration" in sps.columns

        # Sequence 1: A(2), B(1), C(1) = 3 spells
        seq1_spells = sps.filter(pl.col("id") == 1)
        assert len(seq1_spells) == 3

    def test_to_dss(self, state_sequence: StateSequence) -> None:
        """Test converting to DSS format."""
        dss = state_sequence.to_dss()

        # Should remove consecutive duplicates
        # Seq 1: A, A, B, C -> A, B, C (3 transitions)
        # Seq 2: A, B, B, C -> A, B, C (3 transitions)
        # Seq 3: B, B, C, D -> B, C, D (3 transitions)
        assert len(dss) == 9

    def test_encode_states(self, state_sequence: StateSequence) -> None:
        """Test encoding states to integers."""
        encoded = state_sequence.encode_states()

        assert len(encoded) == 12
        # Should be integers
        assert encoded.dtype.kind == "i"

    def test_alphabet_inference(self, state_sequence: StateSequence) -> None:
        """Test that alphabet is correctly inferred."""
        assert len(state_sequence.alphabet) == 4
        assert set(state_sequence.alphabet.states) == {"A", "B", "C", "D"}

    def test_alphabet_mismatch_raises(self, simple_sequence_data: pl.DataFrame) -> None:
        """Supplying an alphabet that doesn't cover the data must raise.

        Regression for v0.3.2 hot-fix A2: previously the mismatch was
        silently accepted and only surfaced as cryptic encoding errors
        downstream.
        """
        narrow = Alphabet(states=("A", "B"))  # missing C and D
        with pytest.raises(ValueError, match="Missing from alphabet"):
            StateSequence(simple_sequence_data, alphabet=narrow)

    def test_alphabet_wider_than_data_is_ok(
        self, simple_sequence_data: pl.DataFrame
    ) -> None:
        """An alphabet with extra states not in the data is allowed.

        Users often declare a domain alphabet wider than the observed sample.
        """
        wide = Alphabet(states=("A", "B", "C", "D", "E", "F"))
        seq = StateSequence(simple_sequence_data, alphabet=wide)
        assert set(seq.alphabet.states) == {"A", "B", "C", "D", "E", "F"}

    def test_equality_does_not_raise(
        self, simple_sequence_data: pl.DataFrame
    ) -> None:
        """``s == s`` and ``s1 == s2`` must not raise.

        Regression for a v0.4.0 issue where ``StateSequence`` was decorated
        with ``@dataclass``, which generated an ``__eq__`` that delegated to
        ``polars.DataFrame.__eq__`` and crashed with "the truth value of a
        DataFrame is ambiguous". Identity-based comparison (the default for
        a mutable container with no explicit ``__eq__``) is correct here.
        """
        s1 = StateSequence(simple_sequence_data)
        s2 = StateSequence(simple_sequence_data)
        assert s1 == s1
        assert s1 != s2  # distinct instances are not equal by default


class TestStateSequenceMethods:
    """Tests for new analytical methods on StateSequence (v0.4.0)."""

    def test_state_counts_returns_count_per_state(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3],
                "time": [0, 1, 2, 0, 1, 2, 0, 1],
                "state": ["A", "B", "A", "A", "A", "C", "B", "B"],
            }
        )
        seq = StateSequence(data)
        result = seq.state_counts()

        assert result.columns == ["state", "count"]
        as_dict = {row["state"]: row["count"] for row in result.to_dicts()}
        assert as_dict == {"A": 4, "B": 3, "C": 1}

    def test_state_per_sequence_counts_default(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "state": ["A", "B", "A", "B", "B", "C"],
            }
        )
        seq = StateSequence(data)
        result = seq.state_per_sequence()

        assert result.columns == ["id", "state", "count"]
        by_id = {
            (row["id"], row["state"]): row["count"]
            for row in result.to_dicts()
        }
        assert by_id == {
            (1, "A"): 2,
            (1, "B"): 1,
            (2, "B"): 2,
            (2, "C"): 1,
        }

    def test_state_per_sequence_proportion_mode(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 2, 2],
                "time": [0, 1, 2, 3, 0, 1],
                "state": ["A", "A", "B", "B", "C", "C"],
            }
        )
        seq = StateSequence(data)
        result = seq.state_per_sequence(proportion=True)

        assert result.columns == ["id", "state", "proportion"]
        by_id = {
            (row["id"], row["state"]): row["proportion"]
            for row in result.to_dicts()
        }
        assert by_id[(1, "A")] == pytest.approx(0.5)
        assert by_id[(1, "B")] == pytest.approx(0.5)
        assert by_id[(2, "C")] == pytest.approx(1.0)

    def test_duration_returns_spell_durations(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 3, 0, 1, 2],
                "state": ["A", "A", "B", "B", "C", "C", "C"],
            }
        )
        seq = StateSequence(data)
        result = seq.duration()

        assert result.columns == ["id", "state", "duration"]
        rows = [
            (row["id"], row["state"], row["duration"]) for row in result.to_dicts()
        ]
        assert rows == [
            (1, "A", 2),
            (1, "B", 2),
            (2, "C", 3),
        ]

    def test_total_duration_by_state_aggregates_spell_durations(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 3, 0, 1, 2],
                "state": ["A", "A", "B", "B", "A", "A", "B"],
            }
        )
        seq = StateSequence(data)
        result = seq.total_duration_by_state()

        assert result.columns == ["state", "total_duration"]
        as_dict = {
            row["state"]: row["total_duration"] for row in result.to_dicts()
        }
        assert as_dict == {"A": 4, "B": 3}

    def test_spells_per_sequence_counts_runs(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 2, 2, 2, 3, 3],
                "time": [0, 1, 2, 3, 0, 1, 2, 0, 1],
                "state": ["A", "A", "B", "C", "X", "Y", "X", "Z", "Z"],
            }
        )
        seq = StateSequence(data)
        result = seq.spells_per_sequence()

        assert result.columns == ["id", "n_spells"]
        as_dict = {row["id"]: row["n_spells"] for row in result.to_dicts()}
        assert as_dict == {1: 3, 2: 3, 3: 1}

    def test_span_integer_time(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 1, 5, 10, 12],
                "state": ["A", "B", "A", "C", "C"],
            }
        )
        seq = StateSequence(data)
        result = seq.span()

        assert result.columns == ["id", "first", "last", "span"]
        as_dict = {row["id"]: row for row in result.to_dicts()}
        assert as_dict[1]["first"] == 0
        assert as_dict[1]["last"] == 5
        assert as_dict[1]["span"] == 5
        assert as_dict[2]["first"] == 10
        assert as_dict[2]["last"] == 12
        assert as_dict[2]["span"] == 2

    def test_span_datetime_time(self) -> None:
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 5, tzinfo=UTC),
                    datetime(2026, 1, 10, tzinfo=UTC),
                ],
                "state": ["A", "B", "A"],
            }
        )
        seq = StateSequence(data)
        result = seq.span()

        row = result.to_dicts()[0]
        assert row["id"] == 1
        assert row["first"] == datetime(2026, 1, 1, tzinfo=UTC)
        assert row["last"] == datetime(2026, 1, 10, tzinfo=UTC)
        # span is a polars Duration
        assert row["span"].days == 9


class TestGranularity:
    def test_granularity_in_config(self) -> None:
        """Setting granularity should be stored (now string-only, v0.3.2 A6)."""
        config = SequenceConfig(granularity="1d")
        assert config.granularity == "1d"

    def test_sps_without_granularity(self) -> None:
        """to_sps() without granularity counts observations."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 1],
                "time": [0, 60, 120, 180, 240],
                "state": ["A", "A", "A", "B", "B"],
            }
        )
        config = SequenceConfig()  # no granularity
        alphabet = Alphabet(states=("A", "B"))
        seq = StateSequence(data=data, config=config, alphabet=alphabet)
        sps = seq.to_sps()
        durations = sps["duration"].to_list()
        assert durations[0] == 3  # A: 3 observations
        assert durations[1] == 2  # B: 2 observations

    def test_granularity_truncates_datetime(self) -> None:
        """v0.3.2 hot-fix A6: granularity is now a polars truncate unit.
        Datetime timestamps should round DOWN to the unit boundary before
        any downstream operation sees them."""
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [
                    datetime(2026, 4, 16, 8, 15, tzinfo=UTC),
                    datetime(2026, 4, 16, 23, 59, tzinfo=UTC),
                    datetime(2026, 4, 17, 0, 30, tzinfo=UTC),
                    datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                ],
                "state": ["A", "A", "B", "B"],
            }
        )
        config = SequenceConfig(granularity="1d")
        alphabet = Alphabet(states=("A", "B"))
        seq = StateSequence(data=data, config=config, alphabet=alphabet)
        stored_times = seq.data["time"].to_list()
        # All four timestamps must be rounded down to midnight.
        assert stored_times[0] == datetime(2026, 4, 16, 0, 0, tzinfo=UTC)
        assert stored_times[1] == datetime(2026, 4, 16, 0, 0, tzinfo=UTC)
        assert stored_times[2] == datetime(2026, 4, 17, 0, 0, tzinfo=UTC)
        assert stored_times[3] == datetime(2026, 4, 17, 0, 0, tzinfo=UTC)

    def test_granularity_rejects_non_datetime_column(self) -> None:
        """v0.3.2 hot-fix A6: string granularity on an integer time column
        should raise with a clear error, not silently no-op."""
        data = pl.DataFrame({"id": [1, 1], "time": [0, 10], "state": ["A", "B"]})
        config = SequenceConfig(granularity="1d")
        with pytest.raises(ValueError, match="datetime/date dtype"):
            StateSequence(data=data, config=config)

    def test_sps_with_granularity_counts_rows(self) -> None:
        """v0.3.2 hot-fix A6: with string granularity truncating datetime,
        to_sps duration is the row count per spell (the old
        (end-start)/g + 1 numeric formula was removed)."""
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [
                    datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
                    datetime(2026, 4, 16, 12, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 0, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                ],
                "state": ["A", "A", "B", "B"],
            }
        )
        config = SequenceConfig(granularity="1d")
        alphabet = Alphabet(states=("A", "B"))
        seq = StateSequence(data=data, config=config, alphabet=alphabet)
        sps = seq.to_sps()
        durations = sps["duration"].to_list()
        assert durations[0] == 2  # Two rows in the A-spell (both 2026-04-16)
        assert durations[1] == 2  # Two rows in the B-spell (both 2026-04-17)


class TestFromIntervals:
    """Tests for StateSequence.from_intervals classmethod (v0.4.0)."""

    def test_happy_path_integer_time(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "start": [0, 5, 0, 3],
                "end": [3, 8, 3, 6],
                "state": ["A", "B", "X", "Y"],
            }
        )
        seq = StateSequence.from_intervals(data, time_points=[0, 2, 5, 7])

        assert isinstance(seq, StateSequence)
        rows = seq.data.sort(["id", "time"]).to_dicts()
        assert rows == [
            {"id": 1, "time": 0, "state": "A"},
            {"id": 1, "time": 2, "state": "A"},
            {"id": 1, "time": 5, "state": "B"},
            {"id": 1, "time": 7, "state": "B"},
            {"id": 2, "time": 0, "state": "X"},
            {"id": 2, "time": 2, "state": "X"},
            {"id": 2, "time": 5, "state": "Y"},
        ]

    def test_default_time_points_integer(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 4],
                "end": [3, 6],
                "state": ["A", "B"],
            }
        )
        seq = StateSequence.from_intervals(data)
        times = sorted(seq.data["time"].to_list())
        assert times == [0, 1, 2, 4, 5]

    def test_datetime_with_no_time_points_raises(self) -> None:
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1],
                "start": [datetime(2026, 1, 1, tzinfo=UTC)],
                "end": [datetime(2026, 1, 5, tzinfo=UTC)],
                "state": ["A"],
            }
        )
        with pytest.raises(ValueError, match="explicit time_points"):
            StateSequence.from_intervals(data)

    def test_datetime_with_explicit_time_points(self) -> None:
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 5, tzinfo=UTC)],
                "end": [datetime(2026, 1, 4, tzinfo=UTC), datetime(2026, 1, 8, tzinfo=UTC)],
                "state": ["A", "B"],
            }
        )
        seq = StateSequence.from_intervals(
            data,
            time_points=[
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 6, tzinfo=UTC),
            ],
        )
        rows = seq.data.sort("time").to_dicts()
        assert rows == [
            {"id": 1, "time": datetime(2026, 1, 2, tzinfo=UTC), "state": "A"},
            {"id": 1, "time": datetime(2026, 1, 6, tzinfo=UTC), "state": "B"},
        ]

    def test_missing_required_column_raises(self) -> None:
        data = pl.DataFrame(
            {"id": [1], "start": [0], "state": ["A"]}
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            StateSequence.from_intervals(data, time_points=[0])

    def test_invalid_intervals_raise(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1],
                "start": [5],
                "end": [3],
                "state": ["A"],
            }
        )
        with pytest.raises(ValueError, match="end < start"):
            StateSequence.from_intervals(data, time_points=[0])

    def test_latest_start_tiebreak_preserved(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 4],
                "end": [10, 8],
                "state": ["EARLY", "LATE"],
            }
        )
        seq = StateSequence.from_intervals(data, time_points=[5])
        rows = seq.data.to_dicts()
        assert rows == [{"id": 1, "time": 5, "state": "LATE"}]

    def test_custom_config_renames_output_columns(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 4],
                "end": [3, 6],
                "state": ["A", "B"],
            }
        )
        cfg = SequenceConfig(
            id_column="trajectory", time_column="step", state_column="status"
        )
        seq = StateSequence.from_intervals(data, time_points=[1, 5], config=cfg)

        assert seq.data.columns == ["trajectory", "step", "status"]
        rows = seq.data.sort("step").to_dicts()
        assert rows == [
            {"trajectory": 1, "step": 1, "status": "A"},
            {"trajectory": 1, "step": 5, "status": "B"},
        ]


