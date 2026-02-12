"""Tests for NMS, NMSMST, and SVRspell distance metrics."""

import numpy as np
import pytest

from yasqat.metrics.nms import nms_distance, nmsmst_distance, svrspell_distance


class TestNMS:
    """Tests for NMS distance."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 2])
        assert nms_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert nms_distance(a, b) == pytest.approx(nms_distance(b, a))

    def test_different_sequences(self) -> None:
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        d = nms_distance(a, b)
        assert d > 0.0
        assert d <= 1.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert nms_distance(empty, empty) == 0.0

    def test_one_empty(self) -> None:
        a = np.array([0, 1])
        empty = np.array([], dtype=np.int64)
        assert nms_distance(a, empty) == 1.0


class TestNMSMST:
    """Tests for NMSMST distance."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 2])
        assert nmsmst_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([2, 1, 0])
        assert nmsmst_distance(a, b) == pytest.approx(nmsmst_distance(b, a))

    def test_range(self) -> None:
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        d = nmsmst_distance(a, b)
        assert 0.0 <= d <= 1.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert nmsmst_distance(empty, empty) == 0.0


class TestSVRspell:
    """Tests for SVRspell distance."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 0, 1, 1])
        assert svrspell_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([0, 1, 1])
        assert svrspell_distance(a, b) == pytest.approx(svrspell_distance(b, a))

    def test_different_spell_structure(self) -> None:
        a = np.array([0, 0, 0, 1])  # spell 0:3, 1:1
        b = np.array([0, 1, 1, 1])  # spell 0:1, 1:3
        d = svrspell_distance(a, b)
        assert d > 0.0

    def test_completely_different(self) -> None:
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        d = svrspell_distance(a, b)
        assert d > 0.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert svrspell_distance(empty, empty) == 0.0
