"""Tests for Alphabet class."""

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
        """Test that duplicate states raise an error."""
        with pytest.raises(ValueError, match="unique"):
            Alphabet(states=("A", "B", "A"))

    def test_alphabet_with_labels(self) -> None:
        """Test alphabet with custom labels."""
        alpha = Alphabet(
            states=("A", "B", "C"),
            labels={"A": "First", "B": "Second", "C": "Third"},
        )

        assert alpha.get_label("A") == "First"
        assert alpha.get_label("B") == "Second"
        assert alpha.get_label("C") == "Third"

    def test_alphabet_with_colors(self) -> None:
        """Test alphabet with custom colors."""
        colors = {"A": "#FF0000", "B": "#00FF00", "C": "#0000FF"}
        alpha = Alphabet(states=("A", "B", "C"), colors=colors)

        assert alpha.get_color("A") == "#FF0000"
        assert alpha.get_color("B") == "#00FF00"
        assert alpha.get_color("C") == "#0000FF"

    def test_alphabet_default_colors(self) -> None:
        """Test that default colors are assigned."""
        alpha = Alphabet(states=("A", "B", "C"))

        # Should have colors assigned
        assert alpha.colors is not None
        assert "A" in alpha.colors
        assert "B" in alpha.colors
        assert "C" in alpha.colors

    def test_from_sequence(self) -> None:
        """Test creating alphabet from sequence."""
        states = ["A", "B", "A", "C", "B", "D"]
        alpha = Alphabet.from_sequence(states)

        assert len(alpha) == 4
        assert alpha.states == ("A", "B", "C", "D")  # Preserves first occurrence order

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

    def test_with_labels(self) -> None:
        """Test creating new alphabet with different labels."""
        alpha = Alphabet(states=("A", "B", "C"))
        new_labels = {"A": "Label A", "B": "Label B", "C": "Label C"}
        new_alpha = alpha.with_labels(new_labels)

        assert new_alpha.get_label("A") == "Label A"
        assert alpha.states == new_alpha.states

    def test_iteration(self) -> None:
        """Test iterating over alphabet."""
        alpha = Alphabet(states=("A", "B", "C"))
        states_list = list(alpha)

        assert states_list == ["A", "B", "C"]
