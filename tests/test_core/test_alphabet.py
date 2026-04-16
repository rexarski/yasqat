"""Tests for Alphabet class."""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

from yasqat.core.alphabet import Alphabet


class TestAlphabet:
    """Tests for Alphabet class."""

    def test_create_alphabet(self) -> None:
        """Test creating an alphabet."""
        alpha = Alphabet(states=("A", "B", "C"))

        assert len(alpha) == 3
        assert "A" in alpha
        assert "D" not in alpha
        assert alpha[0] == "A"
        assert alpha[1] == "B"
        assert alpha[2] == "C"

    def test_alphabet_unique_states(self) -> None:
        """Test that duplicate states are deduplicated (not an error)."""
        alpha = Alphabet(states=("A", "B", "A"))
        assert len(alpha) == 2
        assert alpha.states == ("A", "B")

    def test_alphabet_with_colors(self) -> None:
        """Test alphabet with custom colors."""
        colors = {"A": "#FF0000", "B": "#00FF00", "C": "#0000FF"}
        alpha = Alphabet(states=("A", "B", "C"), colors=colors)

        assert alpha.get_color("A") == "#FF0000"
        assert alpha.get_color("B") == "#00FF00"
        assert alpha.get_color("C") == "#0000FF"

    def test_alphabet_default_colors(self) -> None:
        """Test that default colors are valid hex strings and distinct per state."""
        alpha = Alphabet(states=("A", "B", "C"))

        assert alpha.colors is not None
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        color_values = []
        for state in alpha.states:
            color = alpha.get_color(state)
            assert hex_pattern.match(color), (
                f"Color {color!r} for state {state!r} is not valid hex"
            )
            color_values.append(color)
        # Each state should get a distinct color
        assert len(set(color_values)) == len(color_values)

    def test_single_state_alphabet(self) -> None:
        """Alphabet with a single state should work."""
        alpha = Alphabet(states=("X",))
        assert len(alpha) == 1
        assert alpha[0] == "X"
        assert "X" in alpha
        assert alpha.colors is not None
        encoded = alpha.encode(["X", "X"])
        assert encoded.tolist() == [0, 0]
        assert alpha.decode(encoded) == ["X", "X"]

    def test_with_colors_does_not_mutate(self) -> None:
        """with_colors() should return a new object, leaving original unchanged."""
        alpha = Alphabet(states=("A", "B"))
        original_color_a = alpha.get_color("A")
        new_colors = {"A": "#000000", "B": "#FFFFFF"}
        new_alpha = alpha.with_colors(new_colors)

        # New alphabet has the new colors
        assert new_alpha.get_color("A") == "#000000"
        # Original is unchanged
        assert alpha.get_color("A") == original_color_a
        assert alpha is not new_alpha

    def test_from_sequence(self) -> None:
        """Test creating alphabet from sequence."""
        states = ["A", "B", "A", "C", "B", "D"]
        alpha = Alphabet.from_sequence(states)

        assert len(alpha) == 4
        assert alpha.states == ("A", "B", "C", "D")  # Sorted unique states

    def test_from_series(self) -> None:
        """Test creating alphabet from polars Series."""
        series = pl.Series(["B", "A", "C", "A", "B"])
        alpha = Alphabet.from_series(series)

        assert len(alpha) == 3
        assert set(alpha.states) == {"A", "B", "C"}

    def test_encode_decode(self) -> None:
        """Test encoding and decoding states."""
        alpha = Alphabet(states=("A", "B", "C", "D"))
        states = ["A", "B", "B", "C", "D", "A"]

        encoded = alpha.encode(states)
        assert isinstance(encoded, np.ndarray)
        assert encoded.tolist() == [0, 1, 1, 2, 3, 0]

        decoded = alpha.decode(encoded)
        assert decoded == states

    def test_index_of(self) -> None:
        """Test getting index of state."""
        alpha = Alphabet(states=("A", "B", "C"))

        assert alpha.index_of("A") == 0
        assert alpha.index_of("B") == 1
        assert alpha.index_of("C") == 2

    def test_with_colors(self) -> None:
        """Test creating new alphabet with different colors."""
        alpha = Alphabet(states=("A", "B", "C"))
        new_colors = {"A": "#111111", "B": "#222222", "C": "#333333"}
        new_alpha = alpha.with_colors(new_colors)

        assert new_alpha.get_color("A") == "#111111"
        assert alpha.states == new_alpha.states

    def test_iteration(self) -> None:
        """Test iterating over alphabet."""
        alpha = Alphabet(states=("A", "B", "C"))
        states_list = list(alpha)

        assert states_list == ["A", "B", "C"]


class TestAlphabetSimplification:
    """Tests for simplified Alphabet without labels."""

    def test_index_of_clear_error(self) -> None:
        """index_of() should raise a clear ValueError for missing states."""
        a = Alphabet(states=("A", "B", "C"))
        with pytest.raises(ValueError, match="State 'X' not found in alphabet"):
            a.index_of("X")

    def test_get_color_unknown_state_raises(self) -> None:
        """get_color() should raise KeyError for states not in alphabet."""
        a = Alphabet(states=("A", "B", "C"))
        with pytest.raises(KeyError):
            a.get_color("Z")

    def test_decode_accepts_ndarray(self) -> None:
        """decode() should accept np.ndarray input directly."""
        a = Alphabet(states=("A", "B", "C"))
        indices = np.array([0, 1, 2, 1, 0], dtype=np.int32)
        result = a.decode(indices)
        assert result == ["A", "B", "C", "B", "A"]

    def test_states_sorted_on_init(self) -> None:
        """States should be sorted regardless of construction order."""
        a = Alphabet(states=("C", "A", "B"))
        assert a.states == ("A", "B", "C")

    def test_from_sequence_sorts(self) -> None:
        """from_sequence() should produce sorted states."""
        a = Alphabet.from_sequence(["C", "A", "B", "A"])
        assert a.states == ("A", "B", "C")
