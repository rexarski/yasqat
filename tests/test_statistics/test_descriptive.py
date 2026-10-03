"""Tests for descriptive statistics."""

import math

import numpy as np
import polars as pl
import pytest

from yasqat.core.alphabet import Alphabet
from yasqat.core.pool import SequencePool
from yasqat.core.sequence import SequenceConfig, StateSequence
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

    def test_entropy_per_sequence_constant_is_zero(self) -> None:
        """A sequence with only one state should have entropy 0.0."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "state": ["A", "A", "A", "B", "A", "B"],
            }
        )
        pool = SequencePool(data)
        result = longitudinal_entropy(pool, per_sequence=True, normalize=True)
        # Seq 1 has only state A -> entropy must be 0.0
        seq1_entropy = result.filter(pl.col("id") == 1)["entropy"][0]
        assert seq1_entropy == pytest.approx(0.0)
        # Seq 2 has two states equally split -> entropy > 0
        seq2_entropy = result.filter(pl.col("id") == 2)["entropy"][0]
        assert seq2_entropy > 0.0


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


def _abcd_pool() -> SequencePool:
    """AABB, ABAB, AAAA, ABCD over the inferred alphabet {A, B, C, D}."""
    seqs = ["AABB", "ABAB", "AAAA", "ABCD"]
    rows = [
        (sid, t, st)
        for sid, seq in enumerate(seqs, start=1)
        for t, st in enumerate(seq)
    ]
    return SequencePool(
        pl.DataFrame(rows, schema=["id", "time", "state"], orient="row")
    )


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
        """Pin seqici on the shared fixture (alphabet A-D, length 4).

        Each sequence has 2 transitions out of 3 and the state distribution
        (2, 1, 1)/4, so C = sqrt((2/3) * (H / log 4)) with
        H = -(0.5 ln 0.5 + 2 * 0.25 ln 0.25) = 1.0397.
        """
        result = complexity_index(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        expected = math.sqrt((2 / 3) * (1.0397207708399179 / math.log(4)))
        for val in result["complexity"].to_list():
            assert val == pytest.approx(expected)
        assert expected == pytest.approx(0.7071067811865476)

    def test_complexity_matches_seqici_formula(self) -> None:
        """Pin TraMineR seqici values derived by hand over alphabet {A,B,C,D}.

        C = sqrt((transitions / (l - 1)) * (H / log|A|)). Values are computed
        from the published formula (Gabadinho et al. 2010); no R run backs
        them yet (.scratch/issues/20).
        """
        pool = _abcd_pool()
        result = complexity_index(pool, per_sequence=True)
        assert result["complexity"].to_list() == pytest.approx(
            [0.408248290463863, 0.7071067811865476, 0.0, 1.0]
        )

    def test_complexity_single_state_alphabet_is_zero(self) -> None:
        """A one-state alphabet has log|A| = 0; the index is defined as 0."""
        data = pl.DataFrame({"id": [1, 1], "time": [0, 1], "state": ["A", "A"]})
        assert complexity_index(SequencePool(data)) == 0.0


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

        # Single spell: phi = 2 (empty + "A"), zero variance -> log2(2) = 1,
        # the minimum of Elzinga's index (TraMineR seqST returns 1 as well).
        assert turb == 1.0

    def test_turbulence_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Pin seqST on the shared fixture.

        Every sequence has DSS of three distinct states (phi = 8) and spell
        durations (2, 1, 1): t_bar = 4/3, s2 = 2/9, s2_max = 2 * (1/3)^2 = 2/9,
        so T = log2(8 * (2/9 + 1) / (2/9 + 1)) = 3.
        """
        result = turbulence(sequence_pool, per_sequence=True)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["turbulence"].to_list() == pytest.approx([3.0, 3.0, 3.0])

    def test_turbulence_matches_seqst_formula(self) -> None:
        """Pin TraMineR seqST values derived by hand.

        T = log2(phi * (s2_max + 1) / (s2 + 1)), phi counting the empty
        subsequence of the DSS. AABB: phi=4 -> 3; ABAB: phi=12 -> log2(12);
        AAAA: phi=2 -> 1; ABCD: phi=16 -> 4. No R run backs these yet
        (.scratch/issues/20).
        """
        result = turbulence(_abcd_pool(), per_sequence=True)
        assert result["turbulence"].to_list() == pytest.approx(
            [3.0, math.log2(12), 1.0, 4.0]
        )

    def test_turbulence_uses_duration_variance(self) -> None:
        """AAAB: durations (3, 1), t_bar = 2, s2 = 1, s2_max = 1, phi = 4 -> 2."""
        data = pl.DataFrame(
            {"id": [1] * 4, "time": [0, 1, 2, 3], "state": ["A", "A", "A", "B"]}
        )
        assert turbulence(SequencePool(data)) == pytest.approx(2.0)


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

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test per-sequence distribution."""
        dist = state_distribution(sequence_pool, per_sequence=True)
        assert "id" in dist.columns
        assert "state" in dist.columns
        assert "proportion" in dist.columns
        # Each sequence's proportions should sum to 1
        per_seq_sums = dist.group_by("id").agg(
            pl.col("proportion").sum().alias("total")
        )
        for total in per_seq_sums["total"].to_list():
            assert total == pytest.approx(1.0)


class TestMeanTimeInState:
    """Tests for mean time in state."""

    def test_mean_time(self, sequence_pool: SequencePool) -> None:
        """Test mean time calculation with value verification."""
        result = mean_time_in_state(sequence_pool)

        assert "state" in result.columns
        assert "total_time" in result.columns
        assert "mean_time" in result.columns
        # Sequence pool: Seq1=A,A,B,C Seq2=A,B,B,C Seq3=B,B,C,D
        # State A appears 3 times total across 2 sequences -> mean = 3/2 = 1.5
        # (Seq1 has 2 A's, Seq2 has 1 A, Seq3 has 0)
        # But mean_time is total_time / n_sequences_with_state
        state_a = result.filter(pl.col("state") == "A")
        assert len(state_a) == 1
        assert state_a["total_time"][0] == 3  # 2 + 1 + 0 = 3 time units

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        """Test per-sequence time in state."""
        result = mean_time_in_state(sequence_pool, per_sequence=True)
        assert "id" in result.columns
        assert "state" in result.columns
        assert "time_in_state" in result.columns
        assert len(result) > 0


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

    def test_granularity_datetime(self) -> None:
        """v0.3.2 hot-fix B3: granularity is now a polars truncate-unit
        string and requires a datetime time column. Two timestamps within
        the same day should collapse into one bucket."""
        from datetime import UTC, datetime

        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [
                    datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                    datetime(2026, 4, 16, 20, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 8, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 20, 0, tzinfo=UTC),
                    datetime(2026, 4, 16, 9, 0, tzinfo=UTC),
                    datetime(2026, 4, 16, 21, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 9, 0, tzinfo=UTC),
                    datetime(2026, 4, 17, 21, 0, tzinfo=UTC),
                ],
                "state": ["A", "A", "B", "B", "A", "B", "B", "B"],
            }
        )
        pool = SequencePool(data)
        result = modal_states(pool, granularity="1d")
        times = sorted(result["time"].unique().to_list())
        assert times == [
            datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
            datetime(2026, 4, 17, 0, 0, tzinfo=UTC),
        ]

    def test_granularity_rejects_int(self) -> None:
        """v0.3.2 hot-fix B3: integer granularity is no longer accepted."""
        from yasqat.core.pool import SequencePool

        data = pl.DataFrame({"id": [1, 1], "time": [0, 1], "state": ["A", "B"]})
        pool = SequencePool(data)
        with pytest.raises(TypeError, match="must be a string"):
            modal_states(pool, granularity=2)  # type: ignore[arg-type]

    def test_granularity_requires_datetime_column(self) -> None:
        """v0.3.2 hot-fix B3: string granularity on an integer time column
        must raise a clear dtype error, not silently compute nonsense."""
        from yasqat.core.pool import SequencePool

        data = pl.DataFrame({"id": [1, 1], "time": [0, 1], "state": ["A", "B"]})
        pool = SequencePool(data)
        with pytest.raises(ValueError, match="datetime/date"):
            modal_states(pool, granularity="1d")


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
        assert result["n_subsequences"][0] == 2  # "" and "A"

    def test_all_different(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1], "time": [0, 1, 2], "state": ["A", "B", "C"]}
        )
        pool = SequencePool(data)
        result = subsequence_count(pool, per_sequence=True)
        # "", "A","B","C","AB","AC","BC","ABC" = 8
        assert result["n_subsequences"][0] == 8

    def test_all_same(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1], "time": [0, 1, 2], "state": ["A", "A", "A"]}
        )
        pool = SequencePool(data)
        result = subsequence_count(pool, per_sequence=True)
        # DSS is "A": "" and "A" = 2 (TraMineR seqsubsn default DSS=TRUE)
        assert result["n_subsequences"][0] == 2
        assert subsequence_count(pool, dss=False) == 4  # "", A, AA, AAA

    def test_per_sequence(self, sequence_pool: SequencePool) -> None:
        result = subsequence_count(sequence_pool, per_sequence=True)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        # DSS forms ABC / ABC / BCD: 2**3 = 8 subsequences each, empty included.
        assert result["n_subsequences"].to_list() == [8, 8, 8]

    def test_dss_collapses_repeats_first(self) -> None:
        """Default counts over the DSS: AABB -> AB -> "", A, B, AB = 4."""
        data = pl.DataFrame(
            {"id": [1] * 4, "time": [0, 1, 2, 3], "state": ["A", "A", "B", "B"]}
        )
        pool = SequencePool(data)
        assert subsequence_count(pool) == 4
        # Full sequence AABB: "", A, B, AA, AB, BB, AAB, ABB, AABB = 9.
        assert subsequence_count(pool, dss=False) == 9

    def test_aggregate(self, sequence_pool: SequencePool) -> None:
        result = subsequence_count(sequence_pool)
        assert isinstance(result, float)
        assert result == pytest.approx(8.0)


class TestNormalizedTurbulence:
    """Tests for normalized turbulence."""

    def test_single_spell_zero(self) -> None:
        data = pl.DataFrame(
            {"id": [1, 1, 1, 1], "time": [0, 1, 2, 3], "state": ["A", "A", "A", "A"]}
        )
        pool = SequencePool(data)
        # Raw T = 1 is the minimum; TraMineR rescales (T - 1) / (T_max - 1).
        assert normalized_turbulence(pool) == 0.0

    def test_in_range(self, sequence_pool: SequencePool) -> None:
        result = normalized_turbulence(sequence_pool, per_sequence=True)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        # T = 3 each (see TestTurbulence). Reference: length 4 cycling the
        # alphabet A B C D -> phi = 16, T_max = 4 -> (3 - 1) / (4 - 1).
        for val in result["normalized_turbulence"].to_list():
            assert val == pytest.approx(2 / 3)

    def test_matches_seqst_norm_on_abcd_pool(self) -> None:
        """T_max = 4 for the ABCD cycle; AABB 2/3, ABAB (log2 12 - 1)/3, AAAA 0."""
        result = normalized_turbulence(_abcd_pool(), per_sequence=True)
        assert result["normalized_turbulence"].to_list() == pytest.approx(
            [2 / 3, (math.log2(12) - 1) / 3, 0.0, 1.0]
        )

    def test_reference_cycles_a_small_alphabet(self) -> None:
        """Two states, length 4: reference A B A B has phi = 12, T_max = log2 12."""
        data = pl.DataFrame(
            {"id": [1] * 4, "time": [0, 1, 2, 3], "state": ["A", "B", "A", "B"]}
        )
        assert normalized_turbulence(SequencePool(data)) == pytest.approx(1.0)

    def test_all_distinct_states_reach_one(self) -> None:
        data = pl.DataFrame(
            {"id": [1] * 4, "time": [0, 1, 2, 3], "state": ["A", "B", "C", "D"]}
        )
        assert normalized_turbulence(SequencePool(data)) == pytest.approx(1.0)

    def test_aggregate(self, sequence_pool: SequencePool) -> None:
        result = normalized_turbulence(sequence_pool)
        assert isinstance(result, float)


def _make_long_sequence(length: int = 200) -> StateSequence:
    """Create a single long sequence for overflow testing."""
    states = ["A", "B", "C"]
    data = pl.DataFrame(
        {
            "id": [1] * length,
            "time": list(range(length)),
            "state": [states[i % 3] for i in range(length)],
        }
    )
    return StateSequence(
        data=data,
        config=SequenceConfig(),
        alphabet=Alphabet(states=("A", "B", "C")),
    )


class TestSubsequenceCountOverflow:
    def test_use_log_returns_finite(self) -> None:
        """use_log=True should return finite log2 values for long sequences."""
        seq = _make_long_sequence(500)
        result = subsequence_count(seq, use_log=True)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_use_log_per_sequence(self) -> None:
        """use_log with per_sequence should use log2_n_subsequences column."""
        seq = _make_long_sequence(100)
        result = subsequence_count(seq, per_sequence=True, use_log=True)
        assert isinstance(result, pl.DataFrame)
        assert "log2_n_subsequences" in result.columns


class TestSubsequenceCountPattern:
    def test_states_filter(self) -> None:
        """states_filter should restrict to subsequences of given states."""
        data = pl.DataFrame(
            {
                "id": [1] * 6,
                "time": list(range(6)),
                "state": ["A", "B", "A", "C", "B", "A"],
            }
        )
        seq = StateSequence(
            data=data,
            config=SequenceConfig(),
            alphabet=Alphabet(states=("A", "B", "C")),
        )
        total = subsequence_count(seq)
        filtered = subsequence_count(seq, states_filter=["A"])
        assert isinstance(filtered, (int, float))
        assert filtered <= total
        assert filtered > 0

    def test_states_filter_empty_result(self) -> None:
        """Filtering to a state not in sequence should return 0."""
        data = pl.DataFrame(
            {
                "id": [1] * 3,
                "time": [0, 1, 2],
                "state": ["A", "B", "A"],
            }
        )
        seq = StateSequence(
            data=data,
            config=SequenceConfig(),
            alphabet=Alphabet(states=("A", "B", "C")),
        )
        result = subsequence_count(seq, states_filter=["C"])
        assert result == 1  # only the empty subsequence remains
        assert result >= 0.0
