"""End-to-end integration tests for the yasqat analysis workflow."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.clustering import pam_clustering
from yasqat.clustering.quality import silhouette_scores
from yasqat.core.pool import SequencePool
from yasqat.metrics.base import DistanceMatrix
from yasqat.statistics.descriptive import complexity_index, longitudinal_entropy


class TestFullWorkflow:
    """Integration tests covering load → pool → distances → clustering → quality."""

    @pytest.fixture()
    def workflow_data(self) -> pl.DataFrame:
        """Synthetic data with two clearly separable groups."""
        rng = np.random.default_rng(0)
        records = []
        # Group A: sequences mostly in states 0-1
        for seq_id in range(10):
            for t in range(6):
                state = rng.choice(["A", "B"], p=[0.8, 0.2])
                records.append({"id": seq_id, "time": t, "state": state})
        # Group B: sequences mostly in states 2-3
        for seq_id in range(10, 20):
            for t in range(6):
                state = rng.choice(["C", "D"], p=[0.8, 0.2])
                records.append({"id": seq_id, "time": t, "state": state})
        return pl.DataFrame(records)

    def test_pool_to_distance_matrix(self, workflow_data: pl.DataFrame) -> None:
        """SequencePool.compute_distances returns a labelled DistanceMatrix."""
        pool = SequencePool(workflow_data)
        dm = pool.compute_distances(method="hamming")

        assert isinstance(dm, DistanceMatrix)
        assert dm.values.shape == (20, 20)
        assert dm.labels is not None
        assert len(dm.labels) == 20
        assert np.allclose(np.diag(dm.values), 0)
        assert np.allclose(dm.values, dm.values.T)

    def test_label_lookup_on_distance_matrix(self, workflow_data: pl.DataFrame) -> None:
        """DistanceMatrix.get_distance works with sequence ID labels."""
        pool = SequencePool(workflow_data)
        dm = pool.compute_distances(method="hamming")

        # Distance from any sequence to itself must be 0
        for seq_id in pool.sequence_ids:
            assert dm.get_distance(seq_id, seq_id) == pytest.approx(0.0)

    def test_clustering_accepts_distance_matrix(
        self, workflow_data: pl.DataFrame
    ) -> None:
        """pam_clustering accepts a DistanceMatrix directly."""
        pool = SequencePool(workflow_data)
        dm = pool.compute_distances(method="hamming")

        result = pam_clustering(dm, n_clusters=2, sequence_ids=pool.sequence_ids)

        assert result.n_clusters == 2
        assert len(result.labels) == 20
        assert len(result.get_medoid_ids()) == 2

    def test_quality_after_clustering(self, workflow_data: pl.DataFrame) -> None:
        """Silhouette scores are computable from clustering results."""
        pool = SequencePool(workflow_data)
        dm = pool.compute_distances(method="hamming")
        result = pam_clustering(dm, n_clusters=2, sequence_ids=pool.sequence_ids)

        scores = silhouette_scores(dm.values, result.labels)

        assert scores.shape == (20,)
        assert np.all(scores >= -1.0)
        assert np.all(scores <= 1.0)
        # Two well-separated groups should have positive mean silhouette
        assert float(scores.mean()) > 0.0

    def test_statistics_on_pool(self, workflow_data: pl.DataFrame) -> None:
        """Descriptive statistics accept SequencePool and return consistent shapes."""
        pool = SequencePool(workflow_data)

        entropy_mean = longitudinal_entropy(pool, normalize=True, per_sequence=False)
        entropy_df = longitudinal_entropy(pool, normalize=True, per_sequence=True)

        assert isinstance(entropy_mean, float)
        assert 0.0 <= entropy_mean <= 1.0
        assert isinstance(entropy_df, pl.DataFrame)
        assert len(entropy_df) == 20

        complexity = complexity_index(pool, per_sequence=True)
        assert isinstance(complexity, pl.DataFrame)
        assert len(complexity) == 20

    def test_condensed_form_roundtrip(self, workflow_data: pl.DataFrame) -> None:
        """DistanceMatrix.to_condensed / from_condensed round-trips correctly."""
        pool = SequencePool(workflow_data)
        dm = pool.compute_distances(method="hamming")

        condensed = dm.to_condensed()
        dm2 = DistanceMatrix.from_condensed(condensed, labels=dm.labels)

        assert np.allclose(dm.values, dm2.values)
        assert dm2.labels == dm.labels
