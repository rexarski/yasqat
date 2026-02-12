"""Tests for financial journey data generation."""

from yasqat.synthetic.financial import (
    generate_financial_journeys,
    generate_markov_sequences,
    generate_simple_sequences,
)


class TestGenerateFinancialJourneys:
    """Tests for financial journey generation."""

    def test_generates_correct_number_of_users(self) -> None:
        """Test that correct number of users is generated."""
        n_users = 100
        df = generate_financial_journeys(n_users=n_users, seed=42)

        unique_users = df["id"].n_unique()
        # May have slightly fewer due to early churn
        assert unique_users <= n_users
        assert unique_users > n_users * 0.8  # At least 80% should complete

    def test_has_required_columns(self) -> None:
        """Test that output has required columns."""
        df = generate_financial_journeys(n_users=10, seed=42)

        assert "id" in df.columns
        assert "time" in df.columns
        assert "state" in df.columns
        assert "state_category" in df.columns
        assert "date" in df.columns

    def test_time_range(self) -> None:
        """Test that time values are within expected range."""
        df = generate_financial_journeys(n_users=10, seed=42)

        assert df["time"].min() >= 0
        assert df["time"].max() <= 364

    def test_reproducible_with_seed(self) -> None:
        """Test that same seed produces same data."""
        df1 = generate_financial_journeys(n_users=10, seed=42)
        df2 = generate_financial_journeys(n_users=10, seed=42)

        assert df1.equals(df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Test that different seeds produce different data."""
        df1 = generate_financial_journeys(n_users=10, seed=42)
        df2 = generate_financial_journeys(n_users=10, seed=123)

        assert not df1.equals(df2)

    def test_valid_state_categories(self) -> None:
        """Test that state categories are valid."""
        df = generate_financial_journeys(n_users=10, seed=42)

        valid_categories = {"onboarding", "activity", "product", "risk", "engagement"}
        actual_categories = set(df["state_category"].unique().to_list())

        assert actual_categories.issubset(valid_categories)

    def test_sorted_output(self) -> None:
        """Test that output is sorted by id and time."""
        df = generate_financial_journeys(n_users=10, seed=42)

        # Check if sorted
        is_sorted = df.equals(df.sort(["id", "time"]))
        assert is_sorted


class TestGenerateSimpleSequences:
    """Tests for simple sequence generation."""

    def test_correct_shape(self) -> None:
        """Test that output has correct shape."""
        n_sequences = 10
        seq_length = 5
        df = generate_simple_sequences(
            n_sequences=n_sequences,
            sequence_length=seq_length,
            seed=42,
        )

        assert len(df) == n_sequences * seq_length
        assert df["id"].n_unique() == n_sequences

    def test_correct_states(self) -> None:
        """Test that generated states are valid."""
        n_states = 3
        df = generate_simple_sequences(
            n_sequences=5,
            sequence_length=10,
            n_states=n_states,
            seed=42,
        )

        valid_states = {chr(ord("A") + i) for i in range(n_states)}
        actual_states = set(df["state"].unique().to_list())

        assert actual_states.issubset(valid_states)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        df1 = generate_simple_sequences(n_sequences=5, seed=42)
        df2 = generate_simple_sequences(n_sequences=5, seed=42)

        assert df1.equals(df2)


class TestGenerateMarkovSequences:
    """Tests for Markov sequence generation."""

    def test_correct_shape(self) -> None:
        """Test that output has correct shape."""
        n_sequences = 10
        seq_length = 5
        df = generate_markov_sequences(
            n_sequences=n_sequences,
            sequence_length=seq_length,
            seed=42,
        )

        assert len(df) == n_sequences * seq_length

    def test_with_custom_transition_matrix(self) -> None:
        """Test with custom transition matrix."""
        import numpy as np

        # Deterministic transitions: A->B, B->C, C->A
        tm = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        df = generate_markov_sequences(
            n_sequences=1,
            sequence_length=6,
            transition_matrix=tm,
            states=["A", "B", "C"],
            seed=42,
        )

        # With deterministic transitions, pattern should be predictable
        states = df["state"].to_list()

        # Each state should transition to the next in cycle
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            if current == "A":
                assert next_state == "B"
            elif current == "B":
                assert next_state == "C"
            else:  # C
                assert next_state == "A"

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        df1 = generate_markov_sequences(n_sequences=5, seed=42)
        df2 = generate_markov_sequences(n_sequences=5, seed=42)

        assert df1.equals(df2)
