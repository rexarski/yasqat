"""Tests for Chi2 distance."""

import numpy as np

from yasqat.metrics.chi2 import (
    Chi2Metric,
    chi2_distance,
    chi2_distance_weighted,
    state_distribution,
)


class TestChi2Distance:
    """Tests for Chi2 distance."""

    def test_identical_sequences(self) -> None:
        """Test Chi2 of identical sequences."""
        seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

        dist = chi2_distance(seq, seq)

        assert dist == 0.0

    def test_same_distribution(self) -> None:
        """Test Chi2 when sequences have same state distribution."""
        seq_a = np.array([0, 0, 1, 1], dtype=np.int32)
        seq_b = np.array([1, 1, 0, 0], dtype=np.int32)

        dist = chi2_distance(seq_a, seq_b)

        # Same counts (2 of each state), distance should be 0
        assert dist == 0.0

    def test_different_distributions(self) -> None:
        """Test Chi2 with different state distributions."""
        seq_a = np.array([0, 0, 1, 1, 2], dtype=np.int32)  # 2 zeros, 2 ones, 1 two
        seq_b = np.array([0, 1, 1, 2, 2], dtype=np.int32)  # 1 zero, 2 ones, 2 twos

        dist = chi2_distance(seq_a, seq_b)

        # Should be positive
        assert dist > 0

    def test_completely_different_states(self) -> None:
        """Test Chi2 with completely different state sets."""
        seq_a = np.array([0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1], dtype=np.int32)

        dist = chi2_distance(seq_a, seq_b)

        # Chi2 = 0.5 * ((3-0)^2/3 + (0-3)^2/3) = 0.5 * (3 + 3) = 3
        assert dist == 3.0

    def test_normalized(self) -> None:
        """Test normalized Chi2 distance."""
        seq_a = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2, 2], dtype=np.int32)

        dist_raw = chi2_distance(seq_a, seq_b, normalize=False)
        dist_norm = chi2_distance(seq_a, seq_b, normalize=True)

        # Normalized should be smaller
        assert dist_norm < dist_raw

    def test_with_n_states(self) -> None:
        """Test Chi2 with explicit n_states."""
        seq_a = np.array([0, 0], dtype=np.int32)
        seq_b = np.array([0, 1], dtype=np.int32)

        # Without n_states, uses max from sequences
        dist1 = chi2_distance(seq_a, seq_b)

        # With larger n_states (includes unused states)
        dist2 = chi2_distance(seq_a, seq_b, n_states=5)

        # Should give same result (unused states contribute 0)
        assert dist1 == dist2

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = chi2_distance(seq_a, seq_b)

        assert dist == 0.0

    def test_one_empty_sequence(self) -> None:
        """Test with one empty sequence."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = chi2_distance(seq_a, seq_b)

        assert dist == float("inf")

    def test_symmetric(self) -> None:
        """Test that Chi2 is symmetric."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        dist_ab = chi2_distance(seq_a, seq_b)
        dist_ba = chi2_distance(seq_b, seq_a)

        assert dist_ab == dist_ba


class TestChi2DistanceWeighted:
    """Tests for weighted Chi2 distance."""

    def test_uniform_weights(self) -> None:
        """Test that uniform weights give same as unweighted."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        dist_unweighted = chi2_distance(seq_a, seq_b)
        dist_weighted = chi2_distance_weighted(
            seq_a, seq_b, weights=np.ones(3, dtype=np.float64)
        )

        assert abs(dist_unweighted - dist_weighted) < 1e-10

    def test_custom_weights(self) -> None:
        """Test with custom weights."""
        seq_a = np.array([0, 0, 1], dtype=np.int32)  # 2 zeros, 1 one
        seq_b = np.array([0, 1, 1], dtype=np.int32)  # 1 zero, 2 ones

        # Weight state 0 higher
        weights = np.array([2.0, 1.0], dtype=np.float64)

        dist = chi2_distance_weighted(seq_a, seq_b, weights=weights)

        # Should be different from unweighted
        dist_unweighted = chi2_distance(seq_a, seq_b)
        assert dist != dist_unweighted

    def test_zero_weight_ignores_state(self) -> None:
        """Test that zero weight ignores a state."""
        seq_a = np.array([0, 0, 1], dtype=np.int32)
        seq_b = np.array([0, 1, 1], dtype=np.int32)

        # Zero weight for state 0, only compare state 1
        weights = np.array([0.0, 1.0], dtype=np.float64)

        dist = chi2_distance_weighted(seq_a, seq_b, weights=weights)

        # Only state 1 counts: (1-2)^2 / 3 * 0.5 * 1.0 = 1/6
        assert abs(dist - 1 / 6) < 1e-10


class TestStateDistribution:
    """Tests for state distribution function."""

    def test_uniform_distribution(self) -> None:
        """Test distribution with equal state counts."""
        seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

        dist = state_distribution(seq)

        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(dist, expected)

    def test_skewed_distribution(self) -> None:
        """Test distribution with skewed counts."""
        seq = np.array([0, 0, 0, 1], dtype=np.int32)

        dist = state_distribution(seq)

        expected = np.array([0.75, 0.25])
        np.testing.assert_array_almost_equal(dist, expected)

    def test_single_state(self) -> None:
        """Test distribution with only one state."""
        seq = np.array([0, 0, 0], dtype=np.int32)

        dist = state_distribution(seq)

        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(dist, expected)

    def test_empty_sequence(self) -> None:
        """Test distribution of empty sequence."""
        seq = np.array([], dtype=np.int32)

        dist = state_distribution(seq)

        assert len(dist) == 0

    def test_with_n_states(self) -> None:
        """Test distribution with explicit n_states."""
        seq = np.array([0, 0, 1], dtype=np.int32)

        dist = state_distribution(seq, n_states=5)

        # Should have 5 elements, with zeros for unused states
        assert len(dist) == 5
        assert dist[0] == 2 / 3
        assert dist[1] == 1 / 3
        assert dist[2] == 0.0


class TestChi2Metric:
    """Tests for Chi2Metric class."""

    def test_metric_class(self) -> None:
        """Test Chi2Metric class."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        metric = Chi2Metric()
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0

    def test_metric_normalized(self) -> None:
        """Test Chi2Metric with normalization."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        metric = Chi2Metric(normalize=True)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)

    def test_metric_with_weights(self) -> None:
        """Test Chi2Metric with weights."""
        seq_a = np.array([0, 0, 1], dtype=np.int32)
        seq_b = np.array([0, 1, 1], dtype=np.int32)

        weights = np.array([2.0, 1.0], dtype=np.float64)
        metric = Chi2Metric(weights=weights)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0
