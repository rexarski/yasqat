"""Tests for optimal matching distance."""

import numpy as np

from yasqat.metrics.optimal_matching import OptimalMatchingMetric, optimal_matching


class TestOptimalMatching:
    """Tests for optimal matching distance."""

    def test_identical_sequences(self) -> None:
        """Test distance between identical sequences is zero."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        dist = optimal_matching(seq, seq)

        assert dist == 0.0

    def test_single_substitution(self) -> None:
        """Test distance with single substitution."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        # Default sub_cost = 2.0
        dist = optimal_matching(seq_a, seq_b)

        assert dist == 2.0

    def test_single_indel(self) -> None:
        """Test distance with single insertion/deletion."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3], dtype=np.int32)

        # Default indel = 1.0
        dist = optimal_matching(seq_a, seq_b)

        assert dist == 1.0

    def test_custom_indel_cost(self) -> None:
        """Test with custom indel cost."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3], dtype=np.int32)

        dist = optimal_matching(seq_a, seq_b, indel=0.5)

        assert dist == 0.5

    def test_custom_substitution_cost(self) -> None:
        """Test with custom substitution cost."""
        seq_a = np.array([0, 0], dtype=np.int32)
        seq_b = np.array([0, 1], dtype=np.int32)

        dist = optimal_matching(seq_a, seq_b, sub_cost=1.0)

        assert dist == 1.0

    def test_normalized_distance(self) -> None:
        """Test normalized distance."""
        seq_a = np.array([0, 0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1, 1], dtype=np.int32)

        # Unnormalized: 4 substitutions * 2 = 8
        dist = optimal_matching(seq_a, seq_b, normalize=False)
        assert dist == 8.0

        # Normalized by max length (4)
        dist_norm = optimal_matching(seq_a, seq_b, normalize=True)
        assert dist_norm == 2.0

    def test_empty_sequences(self) -> None:
        """Test distance between empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = optimal_matching(seq_a, seq_b)

        assert dist == 0.0

    def test_one_empty_sequence(self) -> None:
        """Test distance when one sequence is empty."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = optimal_matching(seq_a, seq_b)

        # 3 deletions * indel_cost(1.0) = 3.0
        assert dist == 3.0

    def test_symmetric(self, encoded_sequences: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that distance is symmetric."""
        seq_a, seq_b = encoded_sequences

        dist_ab = optimal_matching(seq_a, seq_b)
        dist_ba = optimal_matching(seq_b, seq_a)

        assert dist_ab == dist_ba

    def test_custom_substitution_matrix(self) -> None:
        """Test with custom substitution matrix."""
        seq_a = np.array([0, 1], dtype=np.int32)
        seq_b = np.array([0, 2], dtype=np.int32)

        # Custom matrix where 1->2 costs 0.5
        sm = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.5],
                [1.0, 0.5, 0.0],
            ]
        )

        dist = optimal_matching(seq_a, seq_b, sm=sm)

        assert dist == 0.5

    def test_metric_class(
        self, encoded_sequences: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test OptimalMatchingMetric class."""
        seq_a, seq_b = encoded_sequences

        metric = OptimalMatchingMetric(indel=1.0, sub_cost=2.0)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0
