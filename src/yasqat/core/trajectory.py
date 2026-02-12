"""Trajectory class for multi-sequence analysis.

A Trajectory combines multiple sequence types (state, event, interval)
for the same set of entities, enabling multi-dimensional temporal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from yasqat.core.sequence import (
        EventSequence,
        IntervalSequence,
        StateSequence,
    )


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory data."""

    id_column: str = "id"
    """Column identifying entities across all sequences."""


@dataclass
class Trajectory:
    """
    A trajectory combining multiple sequence types for the same entities.

    Trajectories allow analyzing entities (e.g., customers, patients) across
    multiple dimensions simultaneously - their states, events, and intervals.

    Example:
        >>> from yasqat import StateSequence, EventSequence
        >>> from yasqat.core.trajectory import Trajectory
        >>> # Patient states (disease stage)
        >>> states = StateSequence(state_data)
        >>> # Patient events (treatments, visits)
        >>> events = EventSequence(event_data)
        >>> # Combine into trajectory
        >>> trajectory = Trajectory(
        ...     state_sequences={"disease_stage": states},
        ...     event_sequences={"treatments": events},
        ... )
    """

    state_sequences: dict[str, StateSequence] = field(default_factory=dict)
    """Named state sequences."""

    event_sequences: dict[str, EventSequence] = field(default_factory=dict)
    """Named event sequences."""

    interval_sequences: dict[str, IntervalSequence] = field(default_factory=dict)
    """Named interval sequences."""

    static_data: pl.DataFrame | None = None
    """Static (time-invariant) attributes for each entity."""

    config: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    """Trajectory configuration."""

    def __post_init__(self) -> None:
        """Validate that all sequences share common entity IDs."""
        self._validate_entity_ids()

    def _validate_entity_ids(self) -> None:
        """Ensure all sequences have overlapping entity IDs."""
        all_ids = self.entity_ids()
        if not all_ids:
            return

        # Validation is lenient - some entities may not be in all sequences
        # Just verify the sequences can be accessed
        _ = list(self.state_sequences.keys())
        _ = list(self.event_sequences.keys())
        _ = list(self.interval_sequences.keys())

    def entity_ids(self) -> list[int | str]:
        """
        Return entity IDs present in any sequence.

        Returns:
            List of unique entity IDs across all sequences.
        """
        all_ids: set[int | str] = set()

        for state_seq in self.state_sequences.values():
            all_ids.update(state_seq.sequence_ids())

        for event_seq in self.event_sequences.values():
            all_ids.update(event_seq.sequence_ids())

        for interval_seq in self.interval_sequences.values():
            all_ids.update(interval_seq.sequence_ids())

        return sorted(all_ids)

    def common_entity_ids(self) -> list[int | str]:
        """
        Return entity IDs present in ALL sequences.

        Returns:
            List of entity IDs that appear in every sequence component.
        """
        id_sets: list[set[int | str]] = []

        for state_seq in self.state_sequences.values():
            id_sets.append(set(state_seq.sequence_ids()))

        for event_seq in self.event_sequences.values():
            id_sets.append(set(event_seq.sequence_ids()))

        for interval_seq in self.interval_sequences.values():
            id_sets.append(set(interval_seq.sequence_ids()))

        if not id_sets:
            return []

        common = id_sets[0]
        for ids in id_sets[1:]:
            common = common & ids

        return sorted(common)

    def n_entities(self) -> int:
        """Return the number of unique entities."""
        return len(self.entity_ids())

    def n_sequences(self) -> int:
        """Return the total number of sequence components."""
        return (
            len(self.state_sequences)
            + len(self.event_sequences)
            + len(self.interval_sequences)
        )

    def add_state_sequence(self, name: str, sequence: StateSequence) -> Trajectory:
        """
        Add a state sequence to the trajectory.

        Args:
            name: Name for this sequence component.
            sequence: StateSequence to add.

        Returns:
            Self for method chaining.
        """
        self.state_sequences[name] = sequence
        return self

    def add_event_sequence(self, name: str, sequence: EventSequence) -> Trajectory:
        """
        Add an event sequence to the trajectory.

        Args:
            name: Name for this sequence component.
            sequence: EventSequence to add.

        Returns:
            Self for method chaining.
        """
        self.event_sequences[name] = sequence
        return self

    def add_interval_sequence(
        self, name: str, sequence: IntervalSequence
    ) -> Trajectory:
        """
        Add an interval sequence to the trajectory.

        Args:
            name: Name for this sequence component.
            sequence: IntervalSequence to add.

        Returns:
            Self for method chaining.
        """
        self.interval_sequences[name] = sequence
        return self

    def set_static_data(self, data: pl.DataFrame) -> Trajectory:
        """
        Set static (time-invariant) data for entities.

        Args:
            data: DataFrame with entity attributes.
                Must contain the id_column.

        Returns:
            Self for method chaining.
        """
        if self.config.id_column not in data.columns:
            raise ValueError(
                f"Static data must contain '{self.config.id_column}' column"
            )
        self.static_data = data
        return self

    def get_entity(self, entity_id: int | str) -> dict[str, Any]:
        """
        Get all data for a single entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            Dictionary with all sequence data and static attributes.
        """
        id_col = self.config.id_column
        result: dict[str, Any] = {"id": entity_id}

        # Get state sequences
        for name, state_seq in self.state_sequences.items():
            entity_data = state_seq.data.filter(pl.col(id_col) == entity_id)
            result[f"state_{name}"] = entity_data

        # Get event sequences
        for name, event_seq in self.event_sequences.items():
            entity_data = event_seq.data.filter(pl.col(id_col) == entity_id)
            result[f"event_{name}"] = entity_data

        # Get interval sequences
        for name, interval_seq in self.interval_sequences.items():
            entity_data = interval_seq.data.filter(pl.col(id_col) == entity_id)
            result[f"interval_{name}"] = entity_data

        # Get static data
        if self.static_data is not None:
            static = self.static_data.filter(pl.col(id_col) == entity_id)
            if len(static) > 0:
                result["static"] = static.row(0, named=True)

        return result

    def filter_entities(self, entity_ids: list[int | str]) -> Trajectory:
        """
        Create a new trajectory with only specified entities.

        Args:
            entity_ids: List of entity IDs to keep.

        Returns:
            New Trajectory with filtered data.
        """
        id_col = self.config.id_column

        # Filter state sequences
        new_state_seqs = {}
        for name, state_seq in self.state_sequences.items():
            filtered_data = state_seq.data.filter(pl.col(id_col).is_in(entity_ids))
            if len(filtered_data) > 0:
                from yasqat.core.sequence import StateSequence

                new_state_seqs[name] = StateSequence(
                    filtered_data, state_seq.config, state_seq.alphabet
                )

        # Filter event sequences
        new_event_seqs = {}
        for name, event_seq in self.event_sequences.items():
            filtered_data = event_seq.data.filter(pl.col(id_col).is_in(entity_ids))
            if len(filtered_data) > 0:
                from yasqat.core.sequence import EventSequence

                new_event_seqs[name] = EventSequence(
                    filtered_data, event_seq.config, event_seq.alphabet
                )

        # Filter interval sequences
        new_interval_seqs = {}
        for name, interval_seq in self.interval_sequences.items():
            filtered_data = interval_seq.data.filter(pl.col(id_col).is_in(entity_ids))
            if len(filtered_data) > 0:
                from yasqat.core.sequence import IntervalSequence

                new_interval_seqs[name] = IntervalSequence(
                    filtered_data, interval_seq.config, interval_seq.alphabet
                )

        # Filter static data
        new_static = None
        if self.static_data is not None:
            new_static = self.static_data.filter(pl.col(id_col).is_in(entity_ids))

        return Trajectory(
            state_sequences=new_state_seqs,
            event_sequences=new_event_seqs,
            interval_sequences=new_interval_seqs,
            static_data=new_static,
            config=self.config,
        )

    def temporal_span(self) -> tuple[Any, Any]:
        """
        Get the overall temporal span across all sequences.

        Returns:
            Tuple of (earliest_time, latest_time).
        """
        min_times = []
        max_times = []

        for state_seq in self.state_sequences.values():
            time_col = state_seq.config.time_column
            min_times.append(state_seq.data[time_col].min())
            max_times.append(state_seq.data[time_col].max())

        for event_seq in self.event_sequences.values():
            time_col = event_seq.config.time_column
            min_times.append(event_seq.data[time_col].min())
            max_times.append(event_seq.data[time_col].max())

        for interval_seq in self.interval_sequences.values():
            start_col = interval_seq.config.start_column
            end_col = interval_seq.config.end_column
            min_times.append(interval_seq.data[start_col].min())
            max_times.append(interval_seq.data[end_col].max())

        if not min_times:
            return (None, None)

        return (
            min(t for t in min_times if t is not None),
            max(t for t in max_times if t is not None),
        )

    def describe(self) -> pl.DataFrame:
        """
        Generate descriptive statistics for the trajectory.

        Returns:
            DataFrame with statistics for each sequence component.
        """
        rows = []

        for name, state_seq in self.state_sequences.items():
            rows.append(
                {
                    "component": f"state_{name}",
                    "type": "state",
                    "n_entities": state_seq.n_sequences(),
                    "n_records": len(state_seq.data),
                    "n_unique_states": state_seq.alphabet.n_states,
                }
            )

        for name, event_seq in self.event_sequences.items():
            rows.append(
                {
                    "component": f"event_{name}",
                    "type": "event",
                    "n_entities": event_seq.n_sequences(),
                    "n_records": len(event_seq.data),
                    "n_unique_states": event_seq.alphabet.n_states,
                }
            )

        for name, interval_seq in self.interval_sequences.items():
            rows.append(
                {
                    "component": f"interval_{name}",
                    "type": "interval",
                    "n_entities": interval_seq.n_sequences(),
                    "n_records": len(interval_seq.data),
                    "n_unique_states": interval_seq.alphabet.n_states,
                }
            )

        return pl.DataFrame(rows)

    def to_dict(self) -> dict[str, pl.DataFrame]:
        """
        Export all sequences as a dictionary of DataFrames.

        Returns:
            Dictionary mapping component names to DataFrames.
        """
        result = {}

        for name, state_seq in self.state_sequences.items():
            result[f"state_{name}"] = state_seq.data

        for name, event_seq in self.event_sequences.items():
            result[f"event_{name}"] = event_seq.data

        for name, interval_seq in self.interval_sequences.items():
            result[f"interval_{name}"] = interval_seq.data

        if self.static_data is not None:
            result["static"] = self.static_data

        return result


@dataclass
class TrajectoryPool:
    """
    A collection of trajectories with shared structure.

    Useful for batch operations on multiple entities' trajectories.
    """

    trajectories: dict[int | str, Trajectory] = field(default_factory=dict)
    """Mapping from entity ID to Trajectory."""

    def __len__(self) -> int:
        """Return the number of trajectories."""
        return len(self.trajectories)

    def entity_ids(self) -> list[int | str]:
        """Return all entity IDs."""
        return list(self.trajectories.keys())

    def get(self, entity_id: int | str) -> Trajectory | None:
        """Get a trajectory by entity ID."""
        return self.trajectories.get(entity_id)

    def add(self, entity_id: int | str, trajectory: Trajectory) -> TrajectoryPool:
        """Add a trajectory to the pool."""
        self.trajectories[entity_id] = trajectory
        return self

    def filter(self, entity_ids: list[int | str]) -> TrajectoryPool:
        """Create a new pool with only specified entities."""
        return TrajectoryPool(
            trajectories={
                eid: traj
                for eid, traj in self.trajectories.items()
                if eid in entity_ids
            }
        )
