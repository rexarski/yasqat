"""Tests for normative sequence indicators."""

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.normative import (
    badness,
    degradation,
    individual_state_distribution,
    insecurity,
    integration,
    objective_volatility,
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


@pytest.fixture
def volatility_pool() -> SequencePool:
    """Pool over a 3-state alphabet that distinguishes the volatility weight.

    Seq 1 (A,B,A,B) visits 2 of 3 states with 3 transitions; seq 2 (C,C,C,C)
    exists only to widen the alphabet to {A, B, C}.
    """
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3],
            "state": ["A", "B", "A", "B", "C", "C", "C", "C"],
        }
    )
    return SequencePool(data)


class TestIndividualStateDistribution:
    def test_schema_is_long_format(self, normative_pool: SequencePool) -> None:
        result = individual_state_distribution(normative_pool)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["id", "state", "count", "proportion"]
        # 3 sequences x 2 alphabet states (E, U) -> 6 rows, all states present
        assert len(result) == 6

    def test_all_positive_sequence(self, normative_pool: SequencePool) -> None:
        result = individual_state_distribution(normative_pool)
        seq1 = result.filter(pl.col("id") == 1)
        e_row = seq1.filter(pl.col("state") == "E")
        u_row = seq1.filter(pl.col("state") == "U")
        # Seq 1 is E,E,E,E
        assert e_row["count"][0] == 4
        assert e_row["proportion"][0] == pytest.approx(1.0)
        # Unvisited state still appears with zero count
        assert u_row["count"][0] == 0
        assert u_row["proportion"][0] == pytest.approx(0.0)

    def test_alternating_sequence(self, normative_pool: SequencePool) -> None:
        result = individual_state_distribution(normative_pool)
        seq3 = result.filter(pl.col("id") == 3)
        # Seq 3 is E,U,E,U -> 2 each, 0.5 each
        assert seq3.filter(pl.col("state") == "E")["count"][0] == 2
        assert seq3.filter(pl.col("state") == "E")["proportion"][0] == pytest.approx(
            0.5
        )
        assert seq3.filter(pl.col("state") == "U")["proportion"][0] == pytest.approx(
            0.5
        )

    def test_accepts_state_sequence(self, normative_pool: SequencePool) -> None:
        seq = normative_pool.to_state_sequence()
        result = individual_state_distribution(seq)
        assert isinstance(result, pl.DataFrame)
        # Seq 1 (E,E,E,E) still resolves through the coercion seam
        e_row = result.filter((pl.col("id") == 1) & (pl.col("state") == "E"))
        assert e_row["count"][0] == 4


class TestObjectiveVolatility:
    def test_stable_sequence_is_zero(self, normative_pool: SequencePool) -> None:
        result = objective_volatility(normative_pool, per_sequence=True)
        # Seq 1 (E,E,E,E): 1 state visited, 0 transitions -> 0.0
        seq1 = result.filter(pl.col("id") == 1)["objective_volatility"][0]
        assert seq1 == pytest.approx(0.0)

    def test_alternating_sequence_is_one(self, normative_pool: SequencePool) -> None:
        result = objective_volatility(normative_pool, per_sequence=True)
        # Seq 3 (E,U,E,U): visits both of 2 states, 3/3 transitions -> 1.0
        seq3 = result.filter(pl.col("id") == 3)["objective_volatility"][0]
        assert seq3 == pytest.approx(1.0)

    def test_aggregate_mean(self, normative_pool: SequencePool) -> None:
        result = objective_volatility(normative_pool)
        assert isinstance(result, float)
        # Mean of 0.0, 0.0, 1.0
        assert result == pytest.approx(1 / 3)

    def test_weight_balances_coverage_and_transitions(
        self, volatility_pool: SequencePool
    ) -> None:
        # Seq 1 (A,B,A,B) over alphabet {A,B,C}:
        #   pvisited = (2-1)/(3-1) = 0.5 ; ptrans = 3/3 = 1.0
        w_half = objective_volatility(volatility_pool, w=0.5, per_sequence=True)
        seq1_half = w_half.filter(pl.col("id") == 1)["objective_volatility"][0]
        assert seq1_half == pytest.approx(0.75)  # 0.5*0.5 + 0.5*1.0

    def test_weight_one_is_pure_coverage(self, volatility_pool: SequencePool) -> None:
        w_one = objective_volatility(volatility_pool, w=1.0, per_sequence=True)
        seq1 = w_one.filter(pl.col("id") == 1)["objective_volatility"][0]
        assert seq1 == pytest.approx(0.5)  # pvisited only

    def test_weight_zero_is_pure_transitions(
        self, volatility_pool: SequencePool
    ) -> None:
        w_zero = objective_volatility(volatility_pool, w=0.0, per_sequence=True)
        seq1 = w_zero.filter(pl.col("id") == 1)["objective_volatility"][0]
        assert seq1 == pytest.approx(1.0)  # ptrans only

    def test_invalid_weight_raises(self, normative_pool: SequencePool) -> None:
        with pytest.raises(ValueError, match="w must be in"):
            objective_volatility(normative_pool, w=1.5)

    def test_accepts_state_sequence(self, normative_pool: SequencePool) -> None:
        seq = normative_pool.to_state_sequence()
        result = objective_volatility(seq)
        assert isinstance(result, float)
        # Same three sequences -> mean of 0.0, 0.0, 1.0
        assert result == pytest.approx(1 / 3)
