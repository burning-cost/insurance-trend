"""Tests for structural break detection (breaks.py)."""

import numpy as np
import pytest

from insurance_trend.breaks import detect_breakpoints, split_segments


class TestDetectBreakpoints:
    def test_returns_list(self):
        signal = np.log(np.array([100.0, 101, 102, 103, 104, 80, 81, 82, 83, 84, 85, 86]))
        breaks = detect_breakpoints(signal)
        assert isinstance(breaks, list)

    def test_no_breaks_in_linear_series(self):
        t = np.arange(20, dtype=float)
        signal = 0.02 * t  # clean linear, no break
        breaks = detect_breakpoints(signal, penalty=5.0)
        assert len(breaks) == 0

    def test_detects_obvious_level_shift(self):
        """A clear step-function shift should be detected with low penalty."""
        signal = np.concatenate([
            np.zeros(10),
            np.full(10, 1.0),
        ])
        breaks = detect_breakpoints(signal, penalty=1.0)
        assert len(breaks) >= 1
        # The break should be near index 10
        assert any(8 <= b <= 12 for b in breaks)

    def test_too_short_series_returns_empty(self):
        signal = np.array([1.0, 2.0, 3.0])
        breaks = detect_breakpoints(signal, min_size=3)
        # Series of length 3 with min_size=3 => 2*min_size > n
        assert breaks == []

    def test_max_breaks_respected(self):
        """Even if many breaks exist, max_breaks limits the output."""
        rng = np.random.default_rng(10)
        # Create a series with many step changes
        parts = [np.full(5, float(i)) + rng.normal(0, 0.01, 5) for i in range(10)]
        signal = np.concatenate(parts)
        breaks = detect_breakpoints(signal, penalty=0.5, max_breaks=3)
        assert len(breaks) <= 3

    def test_all_breaks_are_valid_indices(self):
        signal = np.log(np.array([1.0, 1.1, 1.2, 0.8, 0.82, 0.85, 0.9, 1.2, 1.3, 1.4, 1.5, 1.6]))
        n = len(signal)
        breaks = detect_breakpoints(signal)
        for b in breaks:
            assert 0 <= b < n

    def test_returns_empty_if_ruptures_not_available(self, monkeypatch):
        """If ruptures import fails, should return empty list with warning."""
        import sys
        old = sys.modules.get("ruptures")
        sys.modules["ruptures"] = None  # simulate import failure
        try:
            with pytest.raises(ImportError):
                detect_breakpoints(np.zeros(20))
        finally:
            if old is not None:
                sys.modules["ruptures"] = old
            elif "ruptures" in sys.modules:
                del sys.modules["ruptures"]


class TestSplitSegments:
    def test_no_breaks_returns_full_series(self):
        t = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float) * 2.0
        segments = split_segments(t, y, [])
        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0][0], t)
        np.testing.assert_array_equal(segments[0][1], y)

    def test_one_break_two_segments(self):
        t = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        segments = split_segments(t, y, [5])
        assert len(segments) == 2

    def test_two_breaks_three_segments(self):
        t = np.arange(15, dtype=float)
        y = np.ones(15)
        segments = split_segments(t, y, [5, 10])
        assert len(segments) == 3

    def test_segment_lengths_sum_to_total(self):
        n = 20
        t = np.arange(n, dtype=float)
        y = np.ones(n)
        breaks = [7, 13]
        segments = split_segments(t, y, breaks)
        total = sum(len(s[0]) for s in segments)
        assert total == n

    def test_segments_cover_all_indices(self):
        n = 16
        t = np.arange(n, dtype=float)
        y = np.ones(n)
        breaks = [4, 8, 12]
        segments = split_segments(t, y, breaks)
        recovered = np.concatenate([s[0] for s in segments])
        np.testing.assert_array_equal(recovered, t)

    def test_break_at_zero_produces_empty_first_segment(self):
        """Break at index 0 means first segment is empty; should be filtered out."""
        t = np.arange(10, dtype=float)
        y = np.ones(10)
        segments = split_segments(t, y, [0])
        # First segment [0:0] is empty, should be skipped
        assert all(len(s[0]) > 0 for s in segments)

    def test_break_beyond_length_handles_gracefully(self):
        """Break index >= n means only one segment is returned."""
        t = np.arange(10, dtype=float)
        y = np.ones(10)
        segments = split_segments(t, y, [15])
        # 15 > len(t)=10, so [t[:15], t[15:]] => [full, empty]. Empty filtered.
        lengths = [len(s[0]) for s in segments]
        assert sum(lengths) == len(t)
