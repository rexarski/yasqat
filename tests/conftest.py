"""Pytest fixtures for yasqat tests."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.alphabet import Alphabet
from yasqat.core.pool import SequencePool
from yasqat.core.sequence import StateSequence


@pytest.fixture
def simple_alphabet() -> Alphabet:
    """Create a simple 4-state alphabet."""
    return Alphabet(
        states=("A", "B", "C", "D"),
        labels={"A": "State A", "B": "State B", "C": "State C", "D": "State D"},
    )


@pytest.fixture
def simple_sequence_data() -> pl.DataFrame:
    """Create simple sequence data for testing."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "state": [
                "A",
                "A",
                "B",
                "C",  # Sequence 1
                "A",
                "B",
                "B",
                "C",  # Sequence 2
                "B",
                "B",
                "C",
                "D",  # Sequence 3
            ],
        }
    )


@pytest.fixture
def state_sequence(simple_sequence_data: pl.DataFrame) -> StateSequence:
    """Create a StateSequence from simple data."""
    return StateSequence(simple_sequence_data)


@pytest.fixture
def sequence_pool(simple_sequence_data: pl.DataFrame) -> SequencePool:
    """Create a SequencePool from simple data."""
    return SequencePool(simple_sequence_data)


@pytest.fixture
def encoded_sequences() -> tuple[np.ndarray, np.ndarray]:
    """Create encoded sequences for metric testing."""
    seq_a = np.array([0, 0, 1, 2], dtype=np.int32)  # A, A, B, C
    seq_b = np.array([0, 1, 1, 2], dtype=np.int32)  # A, B, B, C
    return seq_a, seq_b


@pytest.fixture
def equal_length_sequences() -> tuple[np.ndarray, np.ndarray]:
    """Create equal-length sequences for Hamming distance."""
    seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
    seq_b = np.array([0, 2, 2, 3], dtype=np.int32)
    return seq_a, seq_b


@pytest.fixture
def unequal_length_sequences() -> tuple[np.ndarray, np.ndarray]:
    """Create unequal-length sequences for OM/LCS testing."""
    seq_a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    seq_b = np.array([0, 2, 4], dtype=np.int32)
    return seq_a, seq_b


@pytest.fixture
def large_sequence_data() -> pl.DataFrame:
    """Create larger sequence data for performance testing."""
    np.random.seed(42)
    n_sequences = 100
    sequence_length = 20
    states = ["A", "B", "C", "D", "E"]

    records = []
    for seq_id in range(n_sequences):
        for t in range(sequence_length):
            state = np.random.choice(states)
            records.append(
                {
                    "id": seq_id,
                    "time": t,
                    "state": state,
                }
            )

    return pl.DataFrame(records)
