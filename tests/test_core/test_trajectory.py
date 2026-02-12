"""Tests for Trajectory and TrajectoryPool classes."""

import polars as pl
import pytest

from yasqat.core.sequence import EventSequence, IntervalSequence, StateSequence
from yasqat.core.trajectory import Trajectory, TrajectoryConfig, TrajectoryPool


@pytest.fixture
def state_sequence_data() -> pl.DataFrame:
    """Create state sequence data."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "state": ["A", "B", "C", "A", "A", "B"],
        }
    )


@pytest.fixture
def event_sequence_data() -> pl.DataFrame:
    """Create event sequence data."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2],
            "time": [0, 5, 2],
            "state": ["login", "purchase", "login"],
        }
    )


@pytest.fixture
def interval_sequence_data() -> pl.DataFrame:
    """Create interval sequence data."""
    return pl.DataFrame(
        {
            "id": [1, 1, 2],
            "start": [0, 5, 0],
            "end": [5, 10, 8],
            "state": ["working", "meeting", "working"],
        }
    )


@pytest.fixture
def static_data() -> pl.DataFrame:
    """Create static entity data."""
    return pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [30, 25],
        }
    )


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_create_empty_trajectory(self) -> None:
        """Test creating an empty trajectory."""
        traj = Trajectory()

        assert traj.n_entities() == 0
        assert traj.n_sequences() == 0

    def test_create_with_state_sequence(
        self, state_sequence_data: pl.DataFrame
    ) -> None:
        """Test creating trajectory with state sequence."""
        state_seq = StateSequence(state_sequence_data)
        traj = Trajectory(state_sequences={"status": state_seq})

        assert traj.n_entities() == 2
        assert traj.n_sequences() == 1
        assert 1 in traj.entity_ids()
        assert 2 in traj.entity_ids()

    def test_create_with_multiple_sequences(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test creating trajectory with multiple sequence types."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        assert traj.n_sequences() == 2
        assert traj.n_entities() == 2

    def test_add_sequences_fluent(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test adding sequences with fluent API."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = (
            Trajectory()
            .add_state_sequence("status", state_seq)
            .add_event_sequence("events", event_seq)
        )

        assert traj.n_sequences() == 2

    def test_add_interval_sequence(self, interval_sequence_data: pl.DataFrame) -> None:
        """Test adding interval sequence."""
        interval_seq = IntervalSequence(interval_sequence_data)

        traj = Trajectory().add_interval_sequence("activities", interval_seq)

        assert traj.n_sequences() == 1
        assert "activities" in traj.interval_sequences

    def test_set_static_data(
        self,
        state_sequence_data: pl.DataFrame,
        static_data: pl.DataFrame,
    ) -> None:
        """Test setting static data."""
        state_seq = StateSequence(state_sequence_data)

        traj = Trajectory(state_sequences={"status": state_seq}).set_static_data(
            static_data
        )

        assert traj.static_data is not None
        assert len(traj.static_data) == 2

    def test_static_data_validation(self) -> None:
        """Test that static data must have id column."""
        bad_data = pl.DataFrame({"name": ["Alice"], "age": [30]})

        with pytest.raises(ValueError, match="id"):
            Trajectory().set_static_data(bad_data)

    def test_entity_ids(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test getting entity IDs."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        ids = traj.entity_ids()

        assert 1 in ids
        assert 2 in ids

    def test_common_entity_ids(self) -> None:
        """Test getting common entity IDs."""
        # Sequence 1 has entities 1, 2
        data1 = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "A", "B"],
            }
        )

        # Sequence 2 has entities 2, 3
        data2 = pl.DataFrame(
            {
                "id": [2, 2, 3, 3],
                "time": [0, 1, 0, 1],
                "state": ["X", "Y", "X", "Y"],
            }
        )

        traj = Trajectory(
            state_sequences={
                "seq1": StateSequence(data1),
                "seq2": StateSequence(data2),
            }
        )

        common = traj.common_entity_ids()

        # Only entity 2 is in both
        assert common == [2]

    def test_get_entity(
        self,
        state_sequence_data: pl.DataFrame,
        static_data: pl.DataFrame,
    ) -> None:
        """Test getting data for single entity."""
        state_seq = StateSequence(state_sequence_data)

        traj = Trajectory(state_sequences={"status": state_seq}).set_static_data(
            static_data
        )

        entity_data = traj.get_entity(1)

        assert entity_data["id"] == 1
        assert "state_status" in entity_data
        assert "static" in entity_data
        assert entity_data["static"]["name"] == "Alice"

    def test_filter_entities(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test filtering trajectory by entities."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        filtered = traj.filter_entities([1])

        assert filtered.n_entities() == 1
        assert filtered.entity_ids() == [1]

    def test_temporal_span(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test getting temporal span."""
        state_seq = StateSequence(state_sequence_data)  # time 0-2
        event_seq = EventSequence(event_sequence_data)  # time 0-5

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        start, end = traj.temporal_span()

        assert start == 0
        assert end == 5

    def test_describe(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test generating description."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        desc = traj.describe()

        assert "component" in desc.columns
        assert "type" in desc.columns
        assert "n_entities" in desc.columns
        assert len(desc) == 2

    def test_to_dict(
        self,
        state_sequence_data: pl.DataFrame,
        event_sequence_data: pl.DataFrame,
    ) -> None:
        """Test exporting to dictionary."""
        state_seq = StateSequence(state_sequence_data)
        event_seq = EventSequence(event_sequence_data)

        traj = Trajectory(
            state_sequences={"status": state_seq},
            event_sequences={"events": event_seq},
        )

        data_dict = traj.to_dict()

        assert "state_status" in data_dict
        assert "event_events" in data_dict


class TestTrajectoryPool:
    """Tests for TrajectoryPool class."""

    def test_create_empty_pool(self) -> None:
        """Test creating empty pool."""
        pool = TrajectoryPool()

        assert len(pool) == 0
        assert pool.entity_ids() == []

    def test_add_trajectory(self) -> None:
        """Test adding trajectory to pool."""
        pool = TrajectoryPool()

        traj = Trajectory()
        pool.add(1, traj)

        assert len(pool) == 1
        assert 1 in pool.entity_ids()

    def test_get_trajectory(self) -> None:
        """Test getting trajectory from pool."""
        pool = TrajectoryPool()
        traj = Trajectory()
        pool.add("entity_1", traj)

        retrieved = pool.get("entity_1")

        assert retrieved is traj

    def test_get_missing_returns_none(self) -> None:
        """Test getting missing trajectory returns None."""
        pool = TrajectoryPool()

        assert pool.get("nonexistent") is None

    def test_filter_pool(self) -> None:
        """Test filtering pool by entity IDs."""
        pool = TrajectoryPool()
        pool.add(1, Trajectory())
        pool.add(2, Trajectory())
        pool.add(3, Trajectory())

        filtered = pool.filter([1, 3])

        assert len(filtered) == 2
        assert 1 in filtered.entity_ids()
        assert 3 in filtered.entity_ids()
        assert 2 not in filtered.entity_ids()


class TestTrajectoryConfig:
    """Tests for TrajectoryConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TrajectoryConfig()

        assert config.id_column == "id"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TrajectoryConfig(id_column="entity_id")

        assert config.id_column == "entity_id"
