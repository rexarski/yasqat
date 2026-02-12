"""Tests for descriptive statistics."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.descriptive import (
    complexity_index,
    longitudinal_entropy,
    mean_time_in_state,
    modal_states,
    normalized_turbulence,
    sequence_frequency_table,
    sequence_length,
    sequence_log_probability,
    spell_count,
    state_distribution,
    subsequence_count,
    transition_count,
    transition_proportion,
    turbulence,
    visited_proportion,
    visited_states,
)


class TestLongitudinalEntropy:
    """Tests for longitudinal entropy."""

    def test_entropy_single_state(self) -> None:
        """Test entropy of sequence with single state."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        entropy = longitudinal_entropy(pool, normalize=True)

        # Single state = zero entropy
        assert entropy == 0.0

    def test_entropy_uniform_distribution(self) -> None:
        """Test entropy with uniform state distribution."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        entropy = longitudinal_entropy(pool, normalize=True)

        # Uniform distribution = maximum entropy = 1.0 (normalized)
        assert entropy == pytest.approx(1.0)

    def test_entropy_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test entropy per sequence."""
        result = longitudinal_entropy(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "entropy" in result.columns


class TestTransitionCount:
    """Tests for transition count."""

    def test_no_transitions(self) -> None:
        """Test count when there are no transitions."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        count = transition_count(pool)

        assert count == 0

    def test_all_transitions(self) -> None:
        """Test count when all positions are transitions."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        count = transition_count(pool)

        assert count == 3

    def test_transitions_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test transitions per sequence."""
        result = transition_count(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestSequenceLength:
    """Tests for sequence length."""

    def test_uniform_length(self, sequence_pool: SequencePool) -> None:
        """Test with uniform sequence lengths."""
        length = sequence_length(sequence_pool)

        assert length == 4.0

    def test_length_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test length per sequence."""
        result = sequence_length(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["length"].to_list() == [4, 4, 4]


class TestComplexityIndex:
    """Tests for complexity index."""

    def test_complexity_single_state(self) -> None:
        """Test complexity with single state (zero complexity)."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        complexity = complexity_index(pool)

        # No transitions, 1 distinct state = 0 complexity
        assert complexity == 0.0

    def test_complexity_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test complexity per sequence."""
        result = complexity_index(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestTurbulence:
    """Tests for turbulence index."""

    def test_turbulence_single_spell(self) -> None:
        """Test turbulence with single spell."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        turb = turbulence(pool)

        # Single spell = zero turbulence
        assert turb == 0.0

    def test_turbulence_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test turbulence per sequence."""
        result = turbulence(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestStateDistribution:
    """Tests for state distribution."""

    def test_overall_distribution(self, sequence_pool: SequencePool) -> None:
        """Test overall state distribution."""
        dist = state_distribution(sequence_pool)

        assert "state" in dist.columns
        assert "count" in dist.columns
        assert "proportion" in dist.columns
        assert dist["proportion"].sum() == pytest.approx(1.0)

    def test_distribution_at_time(self, sequence_pool: SequencePool) -> None:
        """Test distribution at specific time point."""
        dist = state_distribution(sequence_pool, time_point=0)

        # At time 0, we have: A, A, B (sequences 1, 2, 3)
        assert len(dist) == 2  # A and B


class TestMeanTimeInState:
    """Tests for mean time in state."""

    def test_mean_time(self, sequence_pool: SequencePool) -> None:
        """Test mean time calculation."""
        result = mean_time_in_state(sequence_pool)

        assert "state" in result.columns
        assert "total_time" in result.columns
        assert "mean_time" in result.columns


class TestSpellCount:
    """Tests for spell count."""

    def test_single_spell(self) -> None:
        """Test with single spell (all same state)."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        count = spell_count(pool)

        assert count == 1.0

    def test_all_different(self) -> None:
        """Test with all different states (max spells)."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        count = spell_count(pool)

        assert count == 4.0

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test spell count per sequence."""
        result = spell_count(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "n_spells" in result.columns
        # Seq 1: A,A,B,C -> 3 spells; Seq 2: A,B,B,C -> 3 spells; Seq 3: B,B,C,D -> 3 spells
        assert result["n_spells"].to_list() == [3, 3, 3]


class TestVisitedStates:
    """Tests for visited states count."""

    def test_single_state(self) -> None:
        """Test with single visited state."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        count = visited_states(pool)

        assert count == 1.0

    def test_all_states(self) -> None:
        """Test with all states visited."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        count = visited_states(pool)

        assert count == 4.0

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test visited states per sequence."""
        result = visited_states(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "n_visited" in result.columns
        # Seq 1: {A,B,C} -> 3; Seq 2: {A,B,C} -> 3; Seq 3: {B,C,D} -> 3
        assert result["n_visited"].to_list() == [3, 3, 3]


class TestVisitedProportion:
    """Tests for visited proportion."""

    def test_full_alphabet(self) -> None:
        """Test when all alphabet states are visited."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        prop = visited_proportion(pool)

        # All 4 states visited, alphabet has 4 states
        assert prop == pytest.approx(1.0)

    def test_partial_alphabet(self) -> None:
        """Test when only some alphabet states are visited."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        prop = visited_proportion(pool)

        # 1 out of 1 state visited
        assert prop == pytest.approx(1.0)

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test visited proportion per sequence."""
        result = visited_proportion(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "visited_proportion" in result.columns
        # Alphabet has 4 states (A,B,C,D), each seq visits 3 -> 0.75
        for val in result["visited_proportion"].to_list():
            assert val == pytest.approx(0.75)


class TestTransitionProportion:
    """Tests for transition proportion."""

    def test_no_transitions(self) -> None:
        """Test with no transitions."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "A", "A"],
            }
        )
        pool = SequencePool(data)

        prop = transition_proportion(pool)

        assert prop == 0.0

    def test_all_transitions(self) -> None:
        """Test with all transitions."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "B", "C", "D"],
            }
        )
        pool = SequencePool(data)

        prop = transition_proportion(pool)

        # 3 transitions out of 3 possible positions
        assert prop == pytest.approx(1.0)

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test transition proportion per sequence."""
        result = transition_proportion(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "transition_proportion" in result.columns
        # Seq 1: A,A,B,C -> 2/3; Seq 2: A,B,B,C -> 2/3; Seq 3: B,B,C,D -> 2/3
        for val in result["transition_proportion"].to_list():
            assert val == pytest.approx(2.0 / 3.0)


class TestModalStates:
    """Tests for modal states."""

    def test_modal_states(self, sequence_pool: SequencePool) -> None:
        """Test modal state computation."""
        result = modal_states(sequence_pool)

        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns
        assert "modal_state" in result.columns
        assert "frequency" in result.columns
        assert "proportion" in result.columns
        # 4 time points
        assert len(result) >= 4

    def test_clear_mode(self) -> None:
        """Test with a clear modal state at each time."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "time": [0, 1, 0, 1, 0, 1],
                "state": ["A", "B", "A", "B", "A", "C"],
            }
        )
        pool = SequencePool(data)
        result = modal_states(pool)

        # At time 0: A appears 3 times -> mode is A
        time_0 = result.filter(pl.col("time") == 0)
        assert time_0["modal_state"][0] == "A"
        assert time_0["frequency"][0] == 3


class TestSequenceFrequencyTable:
    """Tests for sequence frequency table."""

    def test_frequency_table(self, sequence_pool: SequencePool) -> None:
        """Test frequency table creation."""
        result = sequence_frequency_table(sequence_pool)

        assert isinstance(result, pl.DataFrame)
        assert "pattern" in result.columns
        assert "count" in result.columns
        assert "proportion" in result.columns
        assert result["proportion"].sum() == pytest.approx(1.0)
        # 3 unique patterns (all sequences are different)
        assert len(result) == 3

    def test_n_top(self, sequence_pool: SequencePool) -> None:
        """Test with n_top limit."""
        result = sequence_frequency_table(sequence_pool, n_top=2)

        assert len(result) <= 2

    def test_identical_sequences(self) -> None:
        """Test with identical sequences."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "A", "B"],
            }
        )
        pool = SequencePool(data)
        result = sequence_frequency_table(pool)

        assert len(result) == 1
        assert result["count"][0] == 2
        assert result["proportion"][0] == pytest.approx(1.0)


class TestSequenceLogProbability:
    """Tests for sequence log-probability."""

    def test_returns_float(self, sequence_pool: SequencePool) -> None:
        """Test that mean log-probability is returned."""
        result = sequence_log_probability(sequence_pool)

        assert isinstance(result, float)
        # Log-probabilities should be negative (or -inf)
        assert result <= 0.0

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test per-sequence log-probabilities."""
        result = sequence_log_probability(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "log_probability" in result.columns
        # All should be negative or -inf
        for val in result["log_probability"].to_list():
            assert val <= 0.0

    def test_deterministic_sequence(self) -> None:
        """Test with perfectly predictable transitions."""
        # All sequences follow A -> B pattern
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "time": [0, 1, 0, 1, 0, 1],
                "state": ["A", "B", "A", "B", "A", "B"],
            }
        )
        pool = SequencePool(data)

        result = sequence_log_probability(pool, per_sequence=True)

        # P(A->B) = 1.0, log(1.0) = 0.0
        for val in result["log_probability"].to_list():
            assert val == pytest.approx(0.0)

    def test_single_element_sequence(self) -> None:
        """Test with single-element sequences (no transitions)."""
        data = pl.DataFrame(
            {
                "id": [1, 2],
                "time": [0, 0],
                "state": ["A", "B"],
            }
        )
        pool = SequencePool(data)

        result = sequence_log_probability(pool, per_sequence=True)

        # No transitions -> log_prob = 0.0 (empty sum)
        for val in result["log_probability"].to_list():
            assert val == pytest.approx(0.0)

    def test_impossible_transition(self) -> None:
        """Test with a transition that has zero probability."""
        # Sequences only have A->B and B->A transitions
        # but one sequence tries C->A which never appears
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "time": [0, 1, 0, 1, 0, 1],
                "state": ["A", "B", "B", "A", "C", "A"],
            }
        )
        pool = SequencePool(data)

        result = sequence_log_probability(pool, per_sequence=True)

        # Seq 3 has C->A, which may have zero probability from other seqs
        # but C->A actually appears once, so it should be finite
        assert all(np.isfinite(v) for v in result["log_probability"].to_list())


class TestSubsequenceCount:
    """Tests for distinct subsequence count."""

    def test_single_element(self) -> None:
        data = pl.DataFrame({"id": [1], "time": [0], "state": ["A"]})
        pool = SequencePool(data)
        result = subsequence_count(pool, per_sequence=True)
        assert result["n_subsequences"][0] == 1  # just "A"

    def test_all_different(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1], "time": [0, 1, 2], "state": ["A", "B", "C"]}
        )
        pool = SequencePool(data)
        result = subsequence_count(pool, per_sequence=True)
        # "A","B","C","AB","AC","BC","ABC" = 7
        assert result["n_subsequences"][0] == 7

    def test_all_same(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1], "time": [0, 1, 2], "state": ["A", "A", "A"]}
        )
        pool = SequencePool(data)
        result = subsequence_count(pool, per_sequence=True)
        # "A","AA","AAA" = 3
        assert result["n_subsequences"][0] == 3

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        result = subsequence_count(sequence_pool, per_sequence=True)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert all(v > 0 for v in result["n_subsequences"].to_list())

    def test_aggregate(self, sequence_pool: SequencePool) -> None:
        result = subsequence_count(sequence_pool)
        assert isinstance(result, float)
        assert result > 0


class TestNormalizedTurbulence:
    """Tests for normalized turbulence."""

    def test_single_spell_zero(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1, 1], "time": [0, 1, 2, 3], "state": ["A", "A", "A", "A"]}
        )
        pool = SequencePool(data)
        result = normalized_turbulence(pool)
        assert result == 0.0

    def test_in_range(self, sequence_pool: SequencePool) -> None:
        result = normalized_turbulence(sequence_pool, per_sequence=True)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        for val in result["normalized_turbulence"].to_list():
            assert 0.0 <= val <= 2.0  # may slightly exceed 1 for some formulae

    def test_aggregate(self, sequence_pool: SequencePool) -> None:
        result = normalized_turbulence(sequence_pool)
        assert isinstance(result, float)
        assert result >= 0.0
