"""Tests for normative sequence indicators."""

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.normative import (
    badness,
    degradation,
    insecurity,
    integration,
    precarity,
    proportion_positive,
    volatility,
)


@pytest.fixture
def normative_pool() -> SequencePool:
    """Pool with clear positive/negative patterns."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "state": [
                "E",
                "E",
                "E",
                "E",  # seq 1: all positive
                "U",
                "U",
                "U",
                "U",  # seq 2: all negative
                "E",
                "U",
                "E",
                "U",  # seq 3: alternating
            ],
        }
    )
    return SequencePool(data)


POSITIVE = {"E"}
NEGATIVE = {"U"}


class TestProportionPositive:
    def test_all_positive(self, normative_pool: SequencePool) -> None:
        result = proportion_positive(normative_pool, POSITIVE, per_sequence=True)
        assert isinstance(result, pl.DataFrame)
        # Seq 1: all E -> 1.0
        seq1 = result.filter(pl.col("id") == 1)["proportion_positive"][0]
        assert seq1 == pytest.approx(1.0)

    def test_all_negative(self, normative_pool: SequencePool) -> None:
        result = proportion_positive(normative_pool, POSITIVE, per_sequence=True)
        seq2 = result.filter(pl.col("id") == 2)["proportion_positive"][0]
        assert seq2 == pytest.approx(0.0)

    def test_aggregate(self, normative_pool: SequencePool) -> None:
        result = proportion_positive(normative_pool, POSITIVE)
        assert isinstance(result, float)
        # Mean of 1.0, 0.0, 0.5 = 0.5
        assert result == pytest.approx(0.5)


class TestVolatility:
    def test_stable_zero(self, normative_pool: SequencePool) -> None:
        result = volatility(normative_pool, POSITIVE, NEGATIVE, per_sequence=True)
        # Seq 1 and 2: no sign changes
        seq1 = result.filter(pl.col("id") == 1)["volatility"][0]
        assert seq1 == pytest.approx(0.0)

    def test_alternating_high(self, normative_pool: SequencePool) -> None:
        result = volatility(normative_pool, POSITIVE, NEGATIVE, per_sequence=True)
        seq3 = result.filter(pl.col("id") == 3)["volatility"][0]
        # Seq 3: E,U,E,U -> signs [+1,-1,+1,-1], 3 sign changes / 3 = 1.0
        assert seq3 == pytest.approx(1.0)


class TestPrecarity:
    def test_all_positive_zero(self, normative_pool: SequencePool) -> None:
        result = precarity(normative_pool, NEGATIVE, per_sequence=True)
        seq1 = result.filter(pl.col("id") == 1)["precarity"][0]
        assert seq1 == pytest.approx(0.0)

    def test_all_negative_high(self, normative_pool: SequencePool) -> None:
        result = precarity(normative_pool, NEGATIVE, per_sequence=True)
        seq2 = result.filter(pl.col("id") == 2)["precarity"][0]
        assert seq2 == pytest.approx(1.0)


class TestInsecurity:
    def test_all_positive_zero(self, normative_pool: SequencePool) -> None:
        result = insecurity(normative_pool, NEGATIVE, per_sequence=True)
        seq1 = result.filter(pl.col("id") == 1)["insecurity"][0]
        assert seq1 == pytest.approx(0.0)

    def test_all_negative(self, normative_pool: SequencePool) -> None:
        result = insecurity(normative_pool, NEGATIVE, per_sequence=True)
        seq2 = result.filter(pl.col("id") == 2)["insecurity"][0]
        assert seq2 == pytest.approx(1.0)


class TestDegradation:
    def test_stable_zero(self, normative_pool: SequencePool) -> None:
        result = degradation(normative_pool, POSITIVE, NEGATIVE, per_sequence=True)
        seq1 = result.filter(pl.col("id") == 1)["degradation"][0]
        assert seq1 == pytest.approx(0.0)

    def test_alternating_nonzero(self, normative_pool: SequencePool) -> None:
        result = degradation(normative_pool, POSITIVE, NEGATIVE, per_sequence=True)
        seq3 = result.filter(pl.col("id") == 3)["degradation"][0]
        assert seq3 == pytest.approx(2 / 3)


class TestBadness:
    def test_all_positive_zero(self, normative_pool: SequencePool) -> None:
        result = badness(normative_pool, NEGATIVE, per_sequence=True)
        seq1 = result.filter(pl.col("id") == 1)["badness"][0]
        assert seq1 == pytest.approx(0.0)

    def test_all_negative_one(self, normative_pool: SequencePool) -> None:
        result = badness(normative_pool, NEGATIVE, per_sequence=True)
        seq2 = result.filter(pl.col("id") == 2)["badness"][0]
        assert seq2 == pytest.approx(1.0)


class TestIntegration:
    def test_all_positive_one(self, normative_pool: SequencePool) -> None:
        result = integration(normative_pool, POSITIVE, per_sequence=True)
        seq1 = result.filter(pl.col("id") == 1)["integration"][0]
        assert seq1 == pytest.approx(1.0)

    def test_all_negative_zero(self, normative_pool: SequencePool) -> None:
        result = integration(normative_pool, POSITIVE, per_sequence=True)
        seq2 = result.filter(pl.col("id") == 2)["integration"][0]
        assert seq2 == pytest.approx(0.0)

    def test_no_positive_states_returns_per_state(
        self, normative_pool: SequencePool
    ) -> None:
        """When positive_states is None, return integration per state."""
        result = integration(normative_pool)
        assert isinstance(result, pl.DataFrame)
        assert "state" in result.columns
        assert "integration" in result.columns
        # Should have one row per unique state in alphabet
        assert len(result) == len(normative_pool.alphabet.states)

    def test_integration_per_state_values(self, normative_pool: SequencePool) -> None:
        """Verify per-state integration values are computed correctly."""
        result = integration(normative_pool)
        # State E: mean integration across seqs (1.0 + 0.0 + ~0.667) / 3
        state_e = result.filter(pl.col("state") == "E")["integration"][0]
        assert state_e > 0.0
        assert state_e < 1.0
        # State U: mean integration across seqs (0.0 + 1.0 + ~0.333) / 3
        state_u = result.filter(pl.col("state") == "U")["integration"][0]
        assert state_u > 0.0
        assert state_u < 1.0
        # E should have higher integration than U (seq3 favors E early)
        assert state_e > state_u
