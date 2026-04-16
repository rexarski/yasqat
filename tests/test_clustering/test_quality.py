"""Tests for cluster quality metrics."""

import numpy as np
import pytest

from yasqat.clustering.quality import (
    cluster_quality,
    distance_to_center,
    pam_range,
    silhouette_score,
    silhouette_scores,
)


@pytest.fixture
def well_separated_clusters() -> tuple[np.ndarray, np.ndarray]:
    """Create well-separated clusters for testing.

    Two clusters: points 0,1 are close; points 2,3 are close.
    The two groups are far apart.
    """
    dist_matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 1.1, 1.0],
            [1.0, 1.1, 0.0, 0.1],
            [1.1, 1.0, 0.1, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return dist_matrix, labels


@pytest.fixture
def poor_clusters() -> tuple[np.ndarray, np.ndarray]:
    """Create poorly-separated clusters (uniform distances)."""
    dist_matrix = np.array(
        [
            [0.0, 0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return dist_matrix, labels


class TestSilhouetteScores:
    """Tests for per-point silhouette scores."""

    def test_well_separated(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test silhouette with well-separated clusters."""
        dist_matrix, labels = well_separated_clusters
        scores = silhouette_scores(dist_matrix, labels)

        assert len(scores) == 4
        # Well-separated clusters should have high scores
        for s in scores:
            assert s > 0.5

    def test_range(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that scores are in [-1, 1] and positive for well-separated clusters."""
        dist_matrix, labels = well_separated_clusters
        scores = silhouette_scores(dist_matrix, labels)

        for s in scores:
            assert -1.0 <= s <= 1.0
        # Well-separated clusters should all have positive silhouette scores
        assert all(s > 0.0 for s in scores)

    def test_single_cluster(self) -> None:
        """Test with single cluster (all same label)."""
        dist_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 0])
        scores = silhouette_scores(dist_matrix, labels)

        # Single cluster: scores should be 0
        np.testing.assert_array_equal(scores, [0.0, 0.0])

    def test_n_equals_k(self) -> None:
        """Test when n_clusters equals n (each point is its own cluster)."""
        dist_matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        labels = np.array([0, 1, 2])
        scores = silhouette_scores(dist_matrix, labels)

        np.testing.assert_array_equal(scores, [0.0, 0.0, 0.0])


class TestSilhouetteScore:
    """Tests for mean silhouette score."""

    def test_well_separated(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test mean silhouette with well-separated clusters."""
        dist_matrix, labels = well_separated_clusters
        score = silhouette_score(dist_matrix, labels)

        assert score > 0.5

    def test_poor_clusters(self, poor_clusters: tuple[np.ndarray, np.ndarray]) -> None:
        """Test mean silhouette with poor clusters."""
        dist_matrix, labels = poor_clusters
        score = silhouette_score(dist_matrix, labels)

        # Uniform distances should give score close to 0
        assert abs(score) < 0.5

    def test_is_mean_of_per_point(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that mean score equals mean of per-point scores."""
        dist_matrix, labels = well_separated_clusters
        mean_score = silhouette_score(dist_matrix, labels)
        per_point = silhouette_scores(dist_matrix, labels)

        assert mean_score == pytest.approx(np.mean(per_point))


class TestClusterQuality:
    """Tests for cluster_quality function."""

    def test_returns_all_metrics(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that all metrics are returned with values in expected ranges."""
        dist_matrix, labels = well_separated_clusters
        metrics = cluster_quality(dist_matrix, labels)

        assert "ASW" in metrics
        assert "PBC" in metrics
        assert "HG" in metrics
        assert "R2" in metrics
        # ASW in [-1, 1], R2 in [0, 1]
        assert -1.0 <= metrics["ASW"] <= 1.0
        assert 0.0 <= metrics["R2"] <= 1.0

    def test_well_separated_metrics(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test metrics with well-separated clusters."""
        dist_matrix, labels = well_separated_clusters
        metrics = cluster_quality(dist_matrix, labels)

        # ASW should be high
        assert metrics["ASW"] > 0.5
        # HG should be positive (larger distance = different cluster)
        assert metrics["HG"] > 0
        # PBC should be negative (same cluster = smaller distance)
        assert metrics["PBC"] < 0
        # R2 should be high
        assert metrics["R2"] > 0.5

    def test_asw_matches_silhouette_score(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that ASW matches silhouette_score function."""
        dist_matrix, labels = well_separated_clusters
        metrics = cluster_quality(dist_matrix, labels)
        asw = silhouette_score(dist_matrix, labels)

        assert metrics["ASW"] == pytest.approx(asw)

    def test_all_values_finite_and_consistent(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that all metric values are finite and internally consistent."""
        dist_matrix, labels = well_separated_clusters
        metrics = cluster_quality(dist_matrix, labels)

        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"
        # ASW from cluster_quality should match standalone silhouette_score
        assert metrics["ASW"] == pytest.approx(silhouette_score(dist_matrix, labels))


class TestDistanceToCenter:
    """Tests for distance_to_center function."""

    def test_output_length(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist_matrix, labels = well_separated_clusters
        distances = distance_to_center(dist_matrix, labels)
        assert len(distances) == 4

    def test_nonnegative(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        dist_matrix, labels = well_separated_clusters
        distances = distance_to_center(dist_matrix, labels)
        assert all(d >= 0.0 for d in distances)

    def test_small_within_cluster(
        self, well_separated_clusters: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Within-cluster distances should be small for well-separated clusters."""
        dist_matrix, labels = well_separated_clusters
        distances = distance_to_center(dist_matrix, labels)
        # Points 0,1 in cluster 0 are 0.1 apart; points 2,3 in cluster 1 are 0.1 apart
        for d in distances:
            assert d == pytest.approx(0.1)

    def test_singleton_cluster(self) -> None:
        """Singleton cluster should have distance 0."""
        dist_matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
        labels = np.array([0, 1, 1])
        distances = distance_to_center(dist_matrix, labels)
        assert distances[0] == 0.0
        assert distances[1] > 0.0


class TestPAMRange:
    """Tests for pam_range function."""

    def test_returns_all_k(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0, 5.0],
                [1.0, 0.0, 5.0, 6.0, 5.0],
                [5.0, 5.0, 0.0, 1.0, 0.5],
                [6.0, 6.0, 1.0, 0.0, 1.5],
                [5.0, 5.0, 0.5, 1.5, 0.0],
            ]
        )
        results = pam_range(dist, k_range=range(2, 5))
        assert set(results.keys()) == {2, 3, 4}

    def test_contains_quality_metrics(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 5.0, 6.0],
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 1.0, 0.0],
            ]
        )
        results = pam_range(dist, k_range=[2])
        assert "ASW" in results[2]
        assert "PBC" in results[2]
        assert "HG" in results[2]
        assert "R2" in results[2]
        assert "total_cost" in results[2]

    def test_default_k_range(self) -> None:
        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 5.0, 6.0],
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 1.0, 0.0],
            ]
        )
        results = pam_range(dist)
        # Default is range(2, min(n, 11)) = range(2, 4) for n=4
        assert set(results.keys()) == {2, 3}

    def test_skips_invalid_k(self) -> None:
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        results = pam_range(dist, k_range=[1, 2, 5])
        # k=1 is <2, k=5 is >=n, so only k=2 would be valid but k=2>=n=2 also skipped
        assert len(results) == 0

    def test_two_tuple_is_inclusive_with_warning(self) -> None:
        """Regression (v0.3.2 hot-fix D2): a 2-tuple is treated as (start, end)
        inclusive and triggers a DeprecationWarning — previously only the two
        endpoints were iterated, silently returning only k=start and k=end.
        """
        import warnings

        dist = np.array(
            [
                [0.0, 1.0, 5.0, 6.0, 5.0],
                [1.0, 0.0, 5.0, 6.0, 5.0],
                [5.0, 5.0, 0.0, 1.0, 0.5],
                [6.0, 6.0, 1.0, 0.0, 1.5],
                [5.0, 5.0, 0.5, 1.5, 0.0],
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = pam_range(dist, k_values=(2, 4))

        assert set(results.keys()) == {2, 3, 4}
        assert any(
            issubclass(w.category, DeprecationWarning) and "2-tuple" in str(w.message)
            for w in caught
        )

    def test_k_range_helper_is_inclusive(self) -> None:
        """k_range(a, b) returns range(a, b+1) — inclusive on both ends."""
        from yasqat.clustering.quality import k_range

        assert list(k_range(2, 5)) == [2, 3, 4, 5]
        assert list(k_range(3, 3)) == [3]

    def test_k_values_and_k_range_together_raise(self) -> None:
        """Passing both ``k_values`` and legacy ``k_range`` is an error."""
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(TypeError, match="both 'k_values' and 'k_range'"):
            pam_range(dist, k_values=[2], k_range=[2])


class TestPamRangeDistanceMatrix:
    def test_accepts_distance_matrix_object(self) -> None:
        """pam_range should accept a DistanceMatrix, not just np.ndarray."""
        from yasqat.metrics.base import DistanceMatrix

        values = np.array(
            [
                [0, 1, 5, 6],
                [1, 0, 5, 6],
                [5, 5, 0, 1],
                [6, 6, 1, 0],
            ],
            dtype=np.float64,
        )
        dm = DistanceMatrix(values=values, labels=[0, 1, 2, 3])
        result = pam_range(dm, k_range=[2, 3])
        assert 2 in result
        assert 3 in result
        assert "ASW" in result[2]
