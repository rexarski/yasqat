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

    def test_no_labels_attribute(self) -> None:
        """Alphabet should not have a labels attribute."""
        a = Alphabet(states=("A", "B", "C"))
        assert not hasattr(a, "labels")

    def test_no_get_label_method(self) -> None:
        """Alphabet should not have a get_label method."""
        a = Alphabet(states=("A", "B", "C"))
        assert not hasattr(a, "get_label")

    def test_no_with_labels_method(self) -> None:
        """Alphabet should not have a with_labels method."""
        a = Alphabet(states=("A", "B", "C"))
        assert not hasattr(a, "with_labels")

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
