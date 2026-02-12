"""Tests for StateSequence, EventSequence, and IntervalSequence classes."""

import polars as pl
import pytest

from yasqat.core.sequence import (
    EventSequence,
    IntervalSequence,
    SequenceConfig,
    StateSequence,
)


class TestStateSequence:
    """Tests for StateSequence class."""

    def test_create_sequence(self, simple_sequence_data: pl.DataFrame) -> None:
        """Test creating a state sequence."""
        seq = StateSequence(simple_sequence_data)

        assert len(seq) == 12
        assert seq.n_sequences() == 3
        assert seq.sequence_ids() == [1, 2, 3]

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


class TestEventSequence:
    """Tests for EventSequence class."""

    def test_create_event_sequence(self) -> None:
        """Test creating an event sequence."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 5, 10, 2, 8],
                "state": ["login", "purchase", "logout", "login", "logout"],
            }
        )

        seq = EventSequence(data)

        assert seq.n_sequences() == 2
        assert len(seq.alphabet) == 3

    def test_event_counts(self) -> None:
        """Test counting events by type."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 2],
                "time": [0, 1, 0, 1, 2],
                "state": ["A", "B", "A", "B", "B"],
            }
        )

        seq = EventSequence(data)
        counts = seq.event_counts()

        assert len(counts) == 2
        assert counts.filter(pl.col("state") == "B")["count"].item() == 3
        assert counts.filter(pl.col("state") == "A")["count"].item() == 2

    def test_events_per_sequence(self) -> None:
        """Test counting events per sequence."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 1, 2, 0, 1],
                "state": ["A", "B", "C", "A", "B"],
            }
        )

        seq = EventSequence(data)
        counts = seq.events_per_sequence()

        assert len(counts) == 2
        assert counts.filter(pl.col("id") == 1)["n_events"].item() == 3
        assert counts.filter(pl.col("id") == 2)["n_events"].item() == 2


class TestIntervalSequence:
    """Tests for IntervalSequence class."""

    def test_create_interval_sequence(self) -> None:
        """Test creating an interval sequence."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "start": [0, 5, 0, 3],
                "end": [5, 10, 3, 8],
                "state": ["working", "meeting", "working", "break"],
            }
        )

        seq = IntervalSequence(data)

        assert seq.n_sequences() == 2
        assert len(seq.alphabet) == 3

    def test_interval_validation(self) -> None:
        """Test that invalid intervals (end < start) raise error."""
        data = pl.DataFrame(
            {
                "id": [1],
                "start": [5],
                "end": [3],  # Invalid: end < start
                "state": ["A"],
            }
        )

        with pytest.raises(ValueError, match="end < start"):
            IntervalSequence(data)

    def test_duration(self) -> None:
        """Test duration computation."""
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 5],
                "end": [5, 15],
                "state": ["A", "B"],
            }
        )

        seq = IntervalSequence(data)
        durations = seq.duration()

        assert "duration" in durations.columns
        # First interval: 5-0=5, Second: 15-5=10
        dur_list = durations.sort("start")["duration"].to_list()
        assert dur_list == [5, 10]

    def test_total_duration_by_state(self) -> None:
        """Test total duration by state."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2],
                "start": [0, 5, 0],
                "end": [5, 10, 10],
                "state": ["A", "A", "B"],
            }
        )

        seq = IntervalSequence(data)
        totals = seq.total_duration_by_state()

        # A: 5 + 5 = 10, B: 10
        a_total = totals.filter(pl.col("state") == "A")["total_duration"].item()
        b_total = totals.filter(pl.col("state") == "B")["total_duration"].item()
        assert a_total == 10
        assert b_total == 10

    def test_intervals_per_sequence(self) -> None:
        """Test counting intervals per sequence."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "start": [0, 5, 10, 0, 5],
                "end": [5, 10, 15, 5, 10],
                "state": ["A", "B", "A", "C", "D"],
            }
        )

        seq = IntervalSequence(data)
        counts = seq.intervals_per_sequence()

        assert counts.filter(pl.col("id") == 1)["n_intervals"].item() == 3
        assert counts.filter(pl.col("id") == 2)["n_intervals"].item() == 2

    def test_overlapping_intervals(self) -> None:
        """Test detecting overlapping intervals."""
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 3],
                "end": [5, 8],  # Overlaps: 3-5
                "state": ["A", "B"],
            }
        )

        seq = IntervalSequence(data)
        overlaps = seq.overlapping_intervals(1)

        assert len(overlaps) == 1  # One pair of overlapping intervals

    def test_no_overlaps(self) -> None:
        """Test non-overlapping intervals."""
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 5],
                "end": [5, 10],  # No overlap
                "state": ["A", "B"],
            }
        )

        seq = IntervalSequence(data)
        overlaps = seq.overlapping_intervals(1)

        assert len(overlaps) == 0

    def test_has_overlaps(self) -> None:
        """Test has_overlaps method."""
        data_with = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 3],
                "end": [5, 8],
                "state": ["A", "B"],
            }
        )

        data_without = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 5],
                "end": [5, 10],
                "state": ["A", "B"],
            }
        )

        assert IntervalSequence(data_with).has_overlaps() is True
        assert IntervalSequence(data_without).has_overlaps() is False

    def test_span(self) -> None:
        """Test getting temporal span."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2],
                "start": [0, 5, 10],
                "end": [5, 15, 20],
                "state": ["A", "B", "C"],
            }
        )

        seq = IntervalSequence(data)
        spans = seq.span()

        assert "first_start" in spans.columns
        assert "last_end" in spans.columns
        assert "span" in spans.columns

        # Sequence 1: 0-15, span=15
        # Sequence 2: 10-20, span=10
        seq1_span = spans.filter(pl.col("id") == 1)["span"].item()
        seq2_span = spans.filter(pl.col("id") == 2)["span"].item()
        assert seq1_span == 15
        assert seq2_span == 10

    def test_to_state_sequence(self) -> None:
        """Test converting to state sequence."""
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 3],
                "end": [3, 6],
                "state": ["A", "B"],
            }
        )

        seq = IntervalSequence(data)
        state_seq = seq.to_state_sequence(time_points=[0, 1, 2, 3, 4, 5])

        assert "time" in state_seq.columns
        assert "state" in state_seq.columns

        # At t=0,1,2: A; at t=3,4,5: B
        states = state_seq.sort("time")["state"].to_list()
        assert states[:3] == ["A", "A", "A"]
        assert states[3:] == ["B", "B", "B"]

    def test_custom_config(self) -> None:
        """Test interval sequence with custom config."""
        data = pl.DataFrame(
            {
                "entity_id": [1, 1],
                "begin": [0, 5],
                "finish": [5, 10],
                "category": ["X", "Y"],
            }
        )

        config = SequenceConfig(
            id_column="entity_id",
            start_column="begin",
            end_column="finish",
            state_column="category",
        )

        seq = IntervalSequence(data, config=config)

        assert seq.n_sequences() == 1
        assert len(seq.alphabet) == 2


class TestTypeConversions:
    """Tests for type conversions between sequence types."""

    def test_state_to_event(self, state_sequence: StateSequence) -> None:
        event_seq = state_sequence.to_event_sequence()
        assert isinstance(event_seq, EventSequence)
        assert event_seq.n_sequences() == state_sequence.n_sequences()

    def test_state_to_interval(self, state_sequence: StateSequence) -> None:
        interval_seq = state_sequence.to_interval_sequence()
        assert isinstance(interval_seq, IntervalSequence)
        assert interval_seq.n_sequences() == state_sequence.n_sequences()

    def test_event_to_state(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "A", "C"],
            }
        )
        event_seq = EventSequence(data)
        state_seq = event_seq.to_state_sequence()
        assert isinstance(state_seq, StateSequence)
        assert state_seq.n_sequences() == 2

    def test_interval_to_event(self) -> None:
        data = pl.DataFrame(
            {
                "id": [1, 1],
                "start": [0, 5],
                "end": [5, 10],
                "state": ["A", "B"],
            }
        )
        interval_seq = IntervalSequence(data)
        event_seq = interval_seq.to_event_sequence()
        assert isinstance(event_seq, EventSequence)
        assert event_seq.n_sequences() == 1

    def test_roundtrip_state_event_state(self, state_sequence: StateSequence) -> None:
        """State -> Event -> State should preserve data."""
        event_seq = state_sequence.to_event_sequence()
        back = event_seq.to_state_sequence()
        assert back.n_sequences() == state_sequence.n_sequences()
        assert len(back) == len(state_sequence)
