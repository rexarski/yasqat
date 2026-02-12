"""Tests for base metric utilities (build_substitution_matrix)."""

import numpy as np
import pytest

from yasqat.metrics.base import build_substitution_matrix


class TestBuildSubstitutionMatrix:
    """Tests for build_substitution_matrix."""

    def test_constant_method(self) -> None:
        """Test constant substitution costs."""
        sm = build_substitution_matrix(4, method="constant", cost=2.0)

        assert sm.shape == (4, 4)
        # Diagonal is zero
        for i in range(4):
            assert sm[i, i] == 0.0
        # Off-diagonal is 2.0
        assert sm[0, 1] == 2.0
        assert sm[2, 3] == 2.0

    def test_constant_custom_cost(self) -> None:
        """Test constant method with custom cost."""
        sm = build_substitution_matrix(3, method="constant", cost=1.5)

        assert sm[0, 1] == 1.5
        assert sm[0, 0] == 0.0

    def test_trate_method(self) -> None:
        """Test transition-rate-based substitution costs."""
        # Uniform transition rates
        rates = np.full((3, 3), 1.0 / 3.0)
        sm = build_substitution_matrix(3, method="trate", transition_rates=rates)

        assert sm.shape == (3, 3)
        for i in range(3):
            assert sm[i, i] == 0.0
        # c(a,b) = 2 - p(a->b) - p(b->a) = 2 - 1/3 - 1/3 = 4/3
        assert sm[0, 1] == pytest.approx(4.0 / 3.0)

    def test_trate_requires_rates(self) -> None:
        """Test that trate method requires transition_rates."""
        with pytest.raises(ValueError, match="transition_rates required"):
            build_substitution_matrix(3, method="trate")

    def test_indels_method(self) -> None:
        """Test indels substitution costs."""
        freq = np.array([0.5, 0.3, 0.2])
        sm = build_substitution_matrix(3, method="indels", state_frequencies=freq)

        assert sm.shape == (3, 3)
        for i in range(3):
            assert sm[i, i] == 0.0
        # c(0,1) = 1/0.5 + 1/0.3 = 2.0 + 3.333...
        assert sm[0, 1] == pytest.approx(1.0 / 0.5 + 1.0 / 0.3)
        # Symmetric
        assert sm[0, 1] == pytest.approx(sm[1, 0])

    def test_indels_requires_frequencies(self) -> None:
        """Test that indels method requires state_frequencies."""
        with pytest.raises(ValueError, match="state_frequencies required"):
            build_substitution_matrix(3, method="indels")

    def test_indelslog_method(self) -> None:
        """Test indelslog substitution costs."""
        freq = np.array([0.5, 0.3, 0.2])
        sm = build_substitution_matrix(3, method="indelslog", state_frequencies=freq)

        assert sm.shape == (3, 3)
        for i in range(3):
            assert sm[i, i] == 0.0
        # c(0,1) = log(1/0.5) + log(1/0.3)
        expected = np.log(1.0 / 0.5) + np.log(1.0 / 0.3)
        assert sm[0, 1] == pytest.approx(expected)
        assert sm[0, 1] == pytest.approx(sm[1, 0])

    def test_indelslog_requires_frequencies(self) -> None:
        """Test that indelslog method requires state_frequencies."""
        with pytest.raises(ValueError, match="state_frequencies required"):
            build_substitution_matrix(3, method="indelslog")

    def test_future_method(self) -> None:
        """Test future substitution costs (chi-squared on next-state distributions)."""
        # Two states with very different transition patterns
        rates = np.array(
            [
                [0.9, 0.1],  # state 0 mostly stays
                [0.1, 0.9],  # state 1 mostly stays
            ]
        )
        sm = build_substitution_matrix(2, method="future", transition_rates=rates)

        assert sm.shape == (2, 2)
        assert sm[0, 0] == 0.0
        assert sm[1, 1] == 0.0
        # Different transition patterns -> positive cost
        assert sm[0, 1] > 0.0
        assert sm[0, 1] == pytest.approx(sm[1, 0])

    def test_future_identical_distributions(self) -> None:
        """Test future method with identical transition distributions."""
        rates = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
        sm = build_substitution_matrix(2, method="future", transition_rates=rates)

        # Identical next-state distributions -> zero cost
        assert sm[0, 1] == pytest.approx(0.0)

    def test_future_requires_rates(self) -> None:
        """Test that future method requires transition_rates."""
        with pytest.raises(ValueError, match="transition_rates required"):
            build_substitution_matrix(2, method="future")

    def test_unknown_method(self) -> None:
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            build_substitution_matrix(3, method="nonexistent")

    def test_features_method(self) -> None:
        """Test features substitution costs (Gower distance)."""
        features = np.array(
            [
                [1.0, 0.0],  # state 0
                [0.0, 1.0],  # state 1
                [1.0, 1.0],  # state 2
            ]
        )
        sm = build_substitution_matrix(3, method="features", state_frequencies=features)

        assert sm.shape == (3, 3)
        for i in range(3):
            assert sm[i, i] == 0.0
        # Symmetric
        assert sm[0, 1] == pytest.approx(sm[1, 0])
        # States 0 and 2 differ in only 1 dimension
        assert sm[0, 2] < sm[0, 1]

    def test_features_requires_frequencies(self) -> None:
        with pytest.raises(ValueError, match="state_frequencies required"):
            build_substitution_matrix(3, method="features")

    def test_features_1d(self) -> None:
        """Test features method with 1D feature vector."""
        features = np.array([0.0, 0.5, 1.0])
        sm = build_substitution_matrix(3, method="features", state_frequencies=features)
        assert sm[0, 2] == pytest.approx(1.0)  # max range
        assert sm[0, 1] == pytest.approx(0.5)

    def test_all_methods_symmetric(self) -> None:
        """Test that all methods produce symmetric matrices."""
        rates = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
        freq = np.array([0.4, 0.35, 0.25])
        features = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        for method, kwargs in [
            ("constant", {}),
            ("trate", {"transition_rates": rates}),
            ("indels", {"state_frequencies": freq}),
            ("indelslog", {"state_frequencies": freq}),
            ("future", {"transition_rates": rates}),
            ("features", {"state_frequencies": features}),
        ]:
            sm = build_substitution_matrix(3, method=method, **kwargs)
            np.testing.assert_array_almost_equal(
                sm, sm.T, err_msg=f"Method '{method}' is not symmetric"
            )
