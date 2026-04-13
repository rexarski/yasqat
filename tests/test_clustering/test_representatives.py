"""Tests for representative sequence extraction."""

import numpy as np
import pytest

from yasqat.clustering.representatives import (
    extract_representatives,
)


@pytest.fixture
def well_separated_dist() -> np.ndarray:
    """Distance matrix with two clear clusters."""
    return np.array(
        [
            [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            [0.1, 0.0, 0.1, 1.1, 1.0, 1.1],
            [0.2, 0.1, 0.0, 1.2, 1.1, 1.0],
            [1.0, 1.1, 1.2, 0.0, 0.1, 0.2],
            [1.1, 1.0, 1.1, 0.1, 0.0, 0.1],
            [1.2, 1.1, 1.0, 0.2, 0.1, 0.0],
        ]
    )


class TestCentralityStrategy:
    """Tests for centrality-based representative selection."""

    def test_returns_correct_count(self, well_separated_dist: np.ndarray) -> None:
        result = extract_representatives(
            well_separated_dist, n_representatives=2, strategy="centrality"
        )
        assert len(result.indices) == 2
        # All indices must be valid (within distance matrix bounds)
        assert all(0 <= idx < 6 for idx in result.indices)
        # Indices should be unique
        assert len(set(result.indices.tolist())) == 2

    def test_selects_central_points(self, well_separated_dist: np.ndarray) -> None:
        result = extract_representatives(
            well_separated_dist, n_representatives=2, strategy="centrality"
        )
        # Points 1 and 4 are most central in each cluster
        assert set(result.indices) == {1, 4}

    def test_scores_are_total_distances(self, well_separated_dist: np.ndarray) -> None:
        result = extract_representatives(
            well_separated_dist, n_representatives=1, strategy="centrality"
        )
        expected_score = well_separated_dist[result.indices[0]].sum()
        assert result.scores[0] == pytest.approx(expected_score)


class TestFrequencyStrategy:
    """Tests for frequency-based representative selection."""

    def test_returns_correct_count(self, well_separated_dist: np.ndarray) -> None:
        result = extract_representatives(
            well_separated_dist, n_representatives=2, strategy="frequency"
        )
        assert len(result.indices) == 2
        assert all(0 <= idx < 6 for idx in result.indices)
        assert len(set(result.indices.tolist())) == 2

    def test_with_labels_selects_per_cluster(
        self, well_separated_dist: np.ndarray
    ) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = extract_representatives(
            well_separated_dist,
            n_representatives=2,
            strategy="frequency",
            labels=labels,
        )
        selected = set(result.indices.tolist())
        # With labels, should pick one from each cluster
        assert selected & {0, 1, 2}, "no representative from cluster 0"
        assert selected & {3, 4, 5}, "no representative from cluster 1"


class TestDensityStrategy:
    """Tests for density-based representative selection."""

    def test_returns_correct_count(self, well_separated_dist: np.ndarray) -> None:
        result = extract_representatives(
            well_separated_dist, n_representatives=3, strategy="density"
        )
        assert len(result.indices) == 3
        assert all(0 <= idx < 6 for idx in result.indices)
        assert len(set(result.indices.tolist())) == 3

    def test_with_labels_selects_per_cluster(
        self, well_separated_dist: np.ndarray
    ) -> None:
        """With labels, should select one dense point from each cluster."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = extract_representatives(
            well_separated_dist,
            n_representatives=2,
            strategy="density",
            labels=labels,
        )
        selected = set(result.indices.tolist())
        assert selected & {0, 1, 2}, "no representative from cluster 0"
        assert selected & {3, 4, 5}, "no representative from cluster 1"

    def test_high_density_selected(self) -> None:
        """Points close together should have higher density."""
        dist = np.array(
            [
                [0.0, 0.1, 0.1, 5.0],
                [0.1, 0.0, 0.1, 5.0],
                [0.1, 0.1, 0.0, 5.0],
                [5.0, 5.0, 5.0, 0.0],
            ]
        )
        result = extract_representatives(dist, n_representatives=1, strategy="density")
        # Points 0,1,2 are in a dense cluster; point 3 is an outlier
        assert result.indices[0] in [0, 1, 2]


class TestPerCluster:
    """Tests for per-cluster representative extraction."""

    def test_representatives_from_each_cluster(
        self, well_separated_dist: np.ndarray
    ) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = extract_representatives(
            well_separated_dist,
            n_representatives=2,
            strategy="centrality",
            labels=labels,
        )
        assert len(result.indices) == 2
        # One from each cluster
        cluster_0 = set(range(3))
        cluster_1 = set(range(3, 6))
        selected = set(result.indices.tolist())
        assert len(selected & cluster_0) >= 1
        assert len(selected & cluster_1) >= 1

    def test_uneven_distribution(self, well_separated_dist: np.ndarray) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = extract_representatives(
            well_separated_dist,
            n_representatives=3,
            strategy="centrality",
            labels=labels,
        )
        assert len(result.indices) == 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_matrix(self) -> None:
        dist = np.zeros((0, 0))
        result = extract_representatives(dist, n_representatives=1)
        assert len(result.indices) == 0

    def test_single_point(self) -> None:
        dist = np.array([[0.0]])
        result = extract_representatives(dist, n_representatives=1)
        assert len(result.indices) == 1
        assert result.indices[0] == 0

    def test_n_representatives_exceeds_n(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = extract_representatives(dist, n_representatives=5)
        assert len(result.indices) == 2

    def test_unknown_strategy_raises(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown strategy"):
            extract_representatives(dist, n_representatives=1, strategy="bogus")


class TestExtractRepresentativesDistanceMatrix:
    def test_accepts_distance_matrix_object(self) -> None:
        """extract_representatives should accept a DistanceMatrix."""
        from yasqat.metrics.base import DistanceMatrix

        values = np.array(
            [
                [0, 1, 4, 5],
                [1, 0, 4, 5],
                [4, 4, 0, 1],
                [5, 5, 1, 0],
            ],
            dtype=np.float64,
        )
        dm = DistanceMatrix(values=values, labels=[0, 1, 2, 3])
        result = extract_representatives(dm, n_representatives=2)
        assert len(result.indices) == 2
