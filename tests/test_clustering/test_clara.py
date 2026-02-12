"""Tests for CLARA clustering."""

import numpy as np
import pytest

from yasqat.clustering.clara import clara_clustering


@pytest.fixture
def two_cluster_dist() -> np.ndarray:
    """Distance matrix with two well-separated clusters."""
    # 6 points: 3 near each other (cluster 0), 3 near each other (cluster 1)
    dist = np.array(
        [
            [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
            [0.1, 0.0, 0.1, 5.1, 5.0, 5.1],
            [0.2, 0.1, 0.0, 5.2, 5.1, 5.0],
            [5.0, 5.1, 5.2, 0.0, 0.1, 0.2],
            [5.1, 5.0, 5.1, 0.1, 0.0, 0.1],
            [5.2, 5.1, 5.0, 0.2, 0.1, 0.0],
        ]
    )
    return dist


class TestCLARAClustering:
    """Tests for clara_clustering function."""

    def test_basic_clustering(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        assert result.n_clusters == 2
        assert len(result.labels) == 6
        # All points should be assigned
        assert set(result.labels) == {0, 1}

    def test_finds_correct_clusters(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        # Points 0,1,2 should be in one cluster, 3,4,5 in another
        assert result.labels[0] == result.labels[1] == result.labels[2]
        assert result.labels[3] == result.labels[4] == result.labels[5]
        assert result.labels[0] != result.labels[3]

    def test_medoid_count(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        assert len(result.medoid_indices) == 2

    def test_medoids_in_range(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        assert all(0 <= m < 6 for m in result.medoid_indices)

    def test_cluster_sizes(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        sizes = result.cluster_sizes()
        assert sum(sizes.values()) == 6

    def test_custom_sample_size(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(
            two_cluster_dist,
            n_clusters=2,
            sample_size=4,
            n_samples=3,
            random_state=42,
        )
        assert result.sample_size == 4
        assert result.n_samples == 3

    def test_total_cost_positive(self, two_cluster_dist: np.ndarray) -> None:
        result = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        assert result.total_cost >= 0.0

    def test_reproducible(self, two_cluster_dist: np.ndarray) -> None:
        r1 = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        r2 = clara_clustering(two_cluster_dist, n_clusters=2, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)
        assert r1.total_cost == r2.total_cost


class TestEdgeCases:
    """Tests for edge cases."""

    def test_n_clusters_equals_n(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = clara_clustering(dist, n_clusters=2, random_state=42)
        assert result.n_clusters == 2
        assert len(result.labels) == 2

    def test_k_exceeds_n_raises(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="cannot exceed"):
            clara_clustering(dist, n_clusters=5)

    def test_non_square_raises(self) -> None:
        dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5]])
        with pytest.raises(ValueError, match="square"):
            clara_clustering(dist, n_clusters=2)

    def test_sample_size_clipped_to_n(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 5.0],
                [5.0, 5.0, 0.0],
            ]
        )
        result = clara_clustering(dist, n_clusters=2, sample_size=100, random_state=42)
        assert result.sample_size == 3
