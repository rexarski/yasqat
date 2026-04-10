"""Alphabet class for managing state vocabularies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


# Default color palette (categorical-friendly)
DEFAULT_COLORS = [
    "#E41A1C",  # Red
    "#377EB8",  # Blue
    "#4DAF4A",  # Green
    "#984EA3",  # Purple
    "#FF7F00",  # Orange
    "#FFFF33",  # Yellow
    "#A65628",  # Brown
    "#F781BF",  # Pink
    "#999999",  # Gray
    "#66C2A5",  # Teal
    "#FC8D62",  # Salmon
    "#8DA0CB",  # Light blue
    "#E78AC3",  # Light pink
    "#A6D854",  # Light green
    "#FFD92F",  # Gold
    "#E5C494",  # Tan
    "#B3B3B3",  # Light gray
]


@dataclass(frozen=True)
class Alphabet:
    """
    State alphabet with optional colors.

    An alphabet defines the set of valid states for a sequence,
    along with an optional color mapping. States are always sorted
    and deduplicated on construction.

    Attributes:
        states: Tuple of state values (unique, sorted).
        colors: Optional mapping from states to color codes.
    """

    states: tuple[str, ...]
    colors: Mapping[str, str] | None = field(default=None)

    def __post_init__(self) -> None:
        """Sort and deduplicate states, then set default colors if not provided."""
        sorted_unique = tuple(sorted(set(self.states)))
        object.__setattr__(self, "states", sorted_unique)

        # Set default colors if not provided
        if self.colors is None:
            default_colors = {
                state: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                for i, state in enumerate(self.states)
            }
            object.__setattr__(self, "colors", default_colors)

    def __contains__(self, state: str) -> bool:
        """Check if a state is in the alphabet."""
        return state in self.states

    def __len__(self) -> int:
        """Return the number of states in the alphabet."""
        return len(self.states)

    @property
    def n_states(self) -> int:
        """Return the number of states in the alphabet."""
        return len(self.states)

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Iterate over states."""
        return iter(self.states)

    def __getitem__(self, idx: int) -> str:
        """Get state by index."""
        return self.states[idx]

    @classmethod
    def from_sequence(cls, states: Sequence[str]) -> Alphabet:
        """Create an alphabet from a sequence of states (sorted, deduplicated)."""
        unique_states = tuple(
            dict.fromkeys(states)
        )  # Remove dups, sort happens in __post_init__
        return cls(states=unique_states)

    @classmethod
    def from_series(cls, series: pl.Series) -> Alphabet:
        """Create an alphabet from a polars Series."""
        unique_states = tuple(series.unique().sort().to_list())
        return cls(states=unique_states)

    def index_of(self, state: str) -> int:
        """Get the index of a state in the alphabet."""
        try:
            return self.states.index(state)
        except ValueError:
            raise ValueError(
                f"State '{state}' not found in alphabet: {self.states}"
            ) from None

    def get_color(self, state: str) -> str:
        """Get the color for a state."""
        if self.colors is not None and state in self.colors:
            return self.colors[state]
        raise KeyError(f"State '{state}' not in alphabet: {self.states}")

    def encode(self, states: Sequence[str]) -> np.ndarray:
        """Encode states as integer indices."""
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        return np.array([state_to_idx[s] for s in states], dtype=np.int32)

    def decode(self, indices: np.ndarray) -> list[str]:
        """Decode integer indices back to states."""
        idx_list = indices.tolist() if hasattr(indices, "tolist") else indices
        return [self.states[i] for i in idx_list]

    def with_colors(self, colors: Mapping[str, str]) -> Alphabet:
        """Create a new alphabet with updated colors."""
        return Alphabet(states=self.states, colors=colors)
