"""Tests for Time Warp Edit Distance (TWED)."""

import numpy as np
import pytest

from yasqat.metrics.twed import twed_distance


class TestTWED:
    """Tests for twed_distance function."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 2, 3])
        assert twed_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert twed_distance(a, b) == pytest.approx(twed_distance(b, a))

    def test_triangle_inequality(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([0, 1, 1])
        c = np.array([1, 1, 0])
        d_ab = twed_distance(a, b)
        d_bc = twed_distance(b, c)
        d_ac = twed_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-9

    def test_different_lengths(self) -> None:
        a = np.array([0, 1, 2, 3])
        b = np.array([0, 2])
        d = twed_distance(a, b)
        assert d > 0.0
        assert np.isfinite(d)

    def test_single_element(self) -> None:
        a = np.array([0])
        b = np.array([1])
        d = twed_distance(a, b)
        assert d > 0.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert twed_distance(empty, empty) == 0.0

    def test_one_empty(self) -> None:
        a = np.array([0, 1, 2])
        empty = np.array([], dtype=np.int64)
        assert twed_distance(a, empty) == float("inf")
        assert twed_distance(empty, a) == float("inf")


class TestTWEDParameters:
    """Tests for TWED parameter effects."""

    def test_higher_stiffness_increases_distance(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        d_low = twed_distance(a, b, nu=0.001)
        d_high = twed_distance(a, b, nu=1.0)
        # Higher stiffness should increase or keep distance
        assert d_high >= d_low - 1e-9

    def test_higher_lambda_increases_distance(self) -> None:
        a = np.array([0, 1, 2, 3])
        b = np.array([0, 3])
        d_low = twed_distance(a, b, lmbda=0.1)
        d_high = twed_distance(a, b, lmbda=5.0)
        assert d_high >= d_low - 1e-9

    def test_custom_timestamps(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2])
        ts_a = np.array([0.0, 1.0, 2.0])
        ts_b = np.array([0.0, 5.0, 10.0])
        # With stiffness, different timestamps should produce larger distance
        d_same = twed_distance(a, b, nu=1.0)
        d_diff = twed_distance(a, b, nu=1.0, timestamps_a=ts_a, timestamps_b=ts_b)
        assert d_diff >= d_same - 1e-9

    def test_custom_substitution_matrix(self) -> None:
        a = np.array([0, 1])
        b = np.array([0, 2])
        sm = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 1.0], [5.0, 1.0, 0.0]])
        d = twed_distance(a, b, sm=sm)
        assert d > 0.0

    def test_zero_stiffness(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 1, 2])
        d = twed_distance(a, b, nu=0.0)
        assert d == pytest.approx(0.0)
