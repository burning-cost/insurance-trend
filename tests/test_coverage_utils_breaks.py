"""Additional coverage for _utils.py and breaks.py.

Targets code paths not reached by existing tests:
- quarter_dummies: various n, with and without periods arg
- annual_trend_rate: sign, zero, extreme values, monthly periods_per_year
- periods_to_index: pandas Index, Polars Series, numeric, string inputs
- safe_log: edge cases — single value, all identical, large values
- validate_lengths: single array, mismatched, polars/pandas mix
- to_polars_series: list, numpy, pandas rename
- detect_breakpoints: ruptures exception path, min_size edge cases
- split_segments: unsorted breakpoints, break at last index, single element
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# quarter_dummies
# ---------------------------------------------------------------------------


class TestQuarterDummies:
    def test_shape_n_by_3(self):
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(8)
        assert dm.shape == (8, 3)

    def test_default_start_q1(self):
        """Without periods arg, sequence starts at Q1 (index 0 mod 4 == 0)."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(4)
        # Row 0 -> Q1 indicator should be 1
        assert dm[0, 0] == 1.0  # Q1
        assert dm[0, 1] == 0.0  # Q2
        assert dm[0, 2] == 0.0  # Q3
        # Row 3 -> Q4: all zeros (Q4 is the base)
        assert dm[3, 0] == 0.0
        assert dm[3, 1] == 0.0
        assert dm[3, 2] == 0.0

    def test_explicit_periods(self):
        """Pass explicit period array to shift phase."""
        from insurance_trend._utils import quarter_dummies
        # If periods starts at 1 (Q2), first row should have Q2 indicator=1
        dm = quarter_dummies(4, periods=np.array([1, 2, 3, 0]))
        assert dm[0, 1] == 1.0  # Q2 indicator at index 0

    def test_repeated_cycle(self):
        """8 periods starting at Q1 should have Q1 dummies at 0, 4."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(8)
        assert dm[0, 0] == 1.0  # Q1
        assert dm[4, 0] == 1.0  # Q1 again

    def test_q3_indicator(self):
        """Row at index 2 mod 4 == 2 (Q3) should have third dummy=1."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(8)
        assert dm[2, 2] == 1.0  # Q3

    def test_q4_all_zeros(self):
        """Q4 is the base level — all dummies are zero."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(4)
        assert dm[3, 0] == 0.0
        assert dm[3, 1] == 0.0
        assert dm[3, 2] == 0.0

    def test_single_period(self):
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(1)
        assert dm.shape == (1, 3)
        assert dm[0, 0] == 1.0  # Q1

    def test_values_binary(self):
        """All values must be 0 or 1."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(20)
        assert set(dm.flatten().tolist()).issubset({0.0, 1.0})

    def test_at_most_one_per_row(self):
        """At most one indicator is 1 per row."""
        from insurance_trend._utils import quarter_dummies
        dm = quarter_dummies(16)
        assert np.all(dm.sum(axis=1) <= 1)


# ---------------------------------------------------------------------------
# annual_trend_rate
# ---------------------------------------------------------------------------


class TestAnnualTrendRate:
    def test_zero_beta_gives_zero_rate(self):
        from insurance_trend._utils import annual_trend_rate
        assert annual_trend_rate(0.0) == pytest.approx(0.0)

    def test_positive_beta_quarterly(self):
        from insurance_trend._utils import annual_trend_rate
        # beta = log(1.02) / 4 per quarter => 2% pa
        beta = np.log(1.02) / 4
        rate = annual_trend_rate(beta, periods_per_year=4)
        assert rate == pytest.approx(0.02, rel=1e-4)

    def test_negative_beta_gives_negative_rate(self):
        from insurance_trend._utils import annual_trend_rate
        beta = -0.01
        rate = annual_trend_rate(beta, periods_per_year=4)
        assert rate < 0.0

    def test_monthly_periods_per_year(self):
        from insurance_trend._utils import annual_trend_rate
        # beta = log(1.05) / 12 per month => ~5% pa
        beta = np.log(1.05) / 12
        rate = annual_trend_rate(beta, periods_per_year=12)
        assert rate == pytest.approx(0.05, rel=1e-4)

    def test_return_type_is_float(self):
        from insurance_trend._utils import annual_trend_rate
        assert isinstance(annual_trend_rate(0.01), float)

    def test_large_positive_beta(self):
        """Large beta should not raise; just returns a very large rate."""
        from insurance_trend._utils import annual_trend_rate
        rate = annual_trend_rate(5.0, periods_per_year=4)
        assert rate > 1.0  # >100% pa


# ---------------------------------------------------------------------------
# periods_to_index
# ---------------------------------------------------------------------------


class TestPeriodsToIndex:
    def test_list_of_strings_returns_arange(self):
        from insurance_trend._utils import periods_to_index
        periods = ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
        idx = periods_to_index(periods)
        np.testing.assert_array_equal(idx, [0.0, 1.0, 2.0, 3.0])

    def test_numeric_array_returns_arange(self):
        from insurance_trend._utils import periods_to_index
        arr = np.array([2020, 2021, 2022], dtype=float)
        idx = periods_to_index(arr)
        assert len(idx) == 3
        np.testing.assert_array_equal(idx, [0.0, 1.0, 2.0])

    def test_polars_series_input(self):
        from insurance_trend._utils import periods_to_index
        s = pl.Series("p", ["2020Q1", "2020Q2", "2021Q1"])
        idx = periods_to_index(s)
        np.testing.assert_array_equal(idx, [0.0, 1.0, 2.0])

    def test_pandas_series_input(self):
        from insurance_trend._utils import periods_to_index
        s = pd.Series(["2020Q1", "2020Q2"])
        idx = periods_to_index(s)
        np.testing.assert_array_equal(idx, [0.0, 1.0])

    def test_pandas_index_input(self):
        from insurance_trend._utils import periods_to_index
        pidx = pd.Index(["2020Q1", "2020Q2", "2020Q3"])
        idx = periods_to_index(pidx)
        np.testing.assert_array_equal(idx, [0.0, 1.0, 2.0])

    def test_length_matches_input(self):
        from insurance_trend._utils import periods_to_index
        n = 12
        periods = [f"P{i}" for i in range(n)]
        idx = periods_to_index(periods)
        assert len(idx) == n


# ---------------------------------------------------------------------------
# safe_log
# ---------------------------------------------------------------------------


class TestSafeLog:
    def test_positive_values(self):
        from insurance_trend._utils import safe_log
        y = np.array([1.0, np.e, np.e**2])
        log_y = safe_log(y)
        np.testing.assert_allclose(log_y, [0.0, 1.0, 2.0])

    def test_zero_raises(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="non-positive"):
            safe_log(np.array([1.0, 0.0, 2.0]))

    def test_negative_raises(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="non-positive"):
            safe_log(np.array([1.0, -1.0, 2.0]))

    def test_error_message_includes_count(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="2 non-positive"):
            safe_log(np.array([0.0, -1.0, 1.0]))

    def test_single_value(self):
        from insurance_trend._utils import safe_log
        result = safe_log(np.array([100.0]))
        assert result[0] == pytest.approx(np.log(100.0))

    def test_large_values(self):
        from insurance_trend._utils import safe_log
        y = np.array([1e10, 1e12])
        log_y = safe_log(y)
        np.testing.assert_allclose(log_y, np.log(y))

    def test_custom_label_in_error(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="my_series"):
            safe_log(np.array([0.0]), label="my_series")


# ---------------------------------------------------------------------------
# validate_lengths
# ---------------------------------------------------------------------------


class TestValidateLengths:
    def test_matching_lengths_returns_length(self):
        from insurance_trend._utils import validate_lengths
        n = validate_lengths(a=np.ones(5), b=np.ones(5))
        assert n == 5

    def test_mismatched_raises(self):
        from insurance_trend._utils import validate_lengths
        with pytest.raises(ValueError, match="same length"):
            validate_lengths(a=np.ones(5), b=np.ones(3))

    def test_single_array(self):
        from insurance_trend._utils import validate_lengths
        n = validate_lengths(x=np.arange(10))
        assert n == 10

    def test_polars_and_numpy_mix(self):
        from insurance_trend._utils import validate_lengths
        n = validate_lengths(
            a=pl.Series("a", [1.0, 2.0, 3.0]),
            b=np.array([4.0, 5.0, 6.0]),
        )
        assert n == 3

    def test_error_message_includes_names(self):
        from insurance_trend._utils import validate_lengths
        with pytest.raises(ValueError, match="claim_counts"):
            validate_lengths(claim_counts=np.ones(4), earned_exposure=np.ones(6))


# ---------------------------------------------------------------------------
# to_polars_series
# ---------------------------------------------------------------------------


class TestToPolarsSeriesExtra:
    def test_list_input(self):
        from insurance_trend._utils import to_polars_series
        s = to_polars_series([1.0, 2.0, 3.0], name="test")
        assert isinstance(s, pl.Series)
        assert s.name == "test"

    def test_numpy_input(self):
        from insurance_trend._utils import to_polars_series
        arr = np.array([10.0, 20.0])
        s = to_polars_series(arr, name="nums")
        assert isinstance(s, pl.Series)
        np.testing.assert_allclose(s.to_numpy(), arr)

    def test_pandas_series_renamed(self):
        from insurance_trend._utils import to_polars_series
        ps = pd.Series([1.0, 2.0, 3.0], name="old_name")
        s = to_polars_series(ps, name="new_name")
        assert s.name == "new_name"

    def test_polars_series_passthrough(self):
        from insurance_trend._utils import to_polars_series
        original = pl.Series("original", [5.0, 6.0])
        s = to_polars_series(original, name="anything")
        # Returns the same series (passthrough for polars)
        assert isinstance(s, pl.Series)


# ---------------------------------------------------------------------------
# to_numpy: invalid type
# ---------------------------------------------------------------------------


class TestToNumpyInvalidType:
    def test_invalid_type_raises_type_error(self):
        from insurance_trend._utils import to_numpy
        with pytest.raises(TypeError, match="pandas Series, Polars Series, list"):
            to_numpy({"a": 1}, name="bad_input")

    def test_integer_list_coerced_to_float(self):
        from insurance_trend._utils import to_numpy
        arr = to_numpy([1, 2, 3], name="ints")
        assert arr.dtype == float


# ---------------------------------------------------------------------------
# detect_breakpoints: additional edge cases
# ---------------------------------------------------------------------------


class TestDetectBreakpointsExtra:
    def test_exactly_2_min_size_segments(self):
        """n == 2 * min_size should work (not return early)."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.zeros(6)
        # With min_size=3, n=6 is exactly at the boundary: 6 == 2*3, not < 2*min_size
        # So it should attempt detection (may return empty if no break)
        breaks = detect_breakpoints(signal, min_size=3, penalty=5.0)
        assert isinstance(breaks, list)

    def test_n_just_below_threshold_returns_empty(self):
        """n = 2*min_size - 1 should return empty immediately."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.zeros(5)
        breaks = detect_breakpoints(signal, min_size=3)
        assert breaks == []

    def test_very_high_penalty_no_breaks(self):
        """A very high penalty should suppress all detections."""
        from insurance_trend.breaks import detect_breakpoints
        rng = np.random.default_rng(50)
        signal = rng.normal(0, 1, 20)
        breaks = detect_breakpoints(signal, penalty=1000.0)
        assert len(breaks) == 0

    def test_ruptures_exception_returns_empty_with_warning(self, monkeypatch):
        """If algo.predict raises, should catch and warn, returning []."""
        from insurance_trend.breaks import detect_breakpoints
        import ruptures as rpt

        def bad_predict(pen):
            raise RuntimeError("Simulated failure")

        class BadAlgo:
            def fit(self, signal):
                return self
            def predict(self, pen):
                raise RuntimeError("Simulated failure")

        original_Pelt = rpt.Pelt

        def mock_Pelt(*args, **kwargs):
            return BadAlgo()

        monkeypatch.setattr(rpt, "Pelt", mock_Pelt)

        signal = np.zeros(20)
        with pytest.warns(UserWarning, match="Breakpoint detection failed"):
            breaks = detect_breakpoints(signal)
        assert breaks == []

    def test_output_excludes_series_length(self):
        """Ruptures appends n as a sentinel; it must be excluded from output."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.concatenate([np.zeros(10), np.full(10, 1.0)])
        n = len(signal)
        breaks = detect_breakpoints(signal, penalty=0.5)
        assert n not in breaks


# ---------------------------------------------------------------------------
# split_segments: additional edge cases
# ---------------------------------------------------------------------------


class TestSplitSegmentsExtra:
    def test_unsorted_breakpoints_are_sorted(self):
        """split_segments should handle unsorted breakpoints gracefully."""
        from insurance_trend.breaks import split_segments
        t = np.arange(15, dtype=float)
        y = np.ones(15)
        # Unsorted: [10, 5] should be treated as [5, 10]
        segs = split_segments(t, y, [10, 5])
        assert len(segs) == 3
        # First segment should be t[0:5]
        np.testing.assert_array_equal(segs[0][0], t[:5])

    def test_break_at_last_index(self):
        """Break at n-1 should produce one full segment and one empty (filtered out)."""
        from insurance_trend.breaks import split_segments
        t = np.arange(10, dtype=float)
        y = np.ones(10)
        segs = split_segments(t, y, [9])
        # [t[0:9], t[9:]] => [9 elements, 1 element] — both non-empty
        total = sum(len(s[0]) for s in segs)
        assert total == 10

    def test_single_element_segments_not_filtered(self):
        """A breakpoint that creates a 1-element segment should still be included."""
        from insurance_trend.breaks import split_segments
        t = np.arange(10, dtype=float)
        y = np.ones(10)
        segs = split_segments(t, y, [1])
        # Segments: t[0:1] (length 1), t[1:] (length 9)
        lengths = [len(s[0]) for s in segs]
        assert 1 in lengths

    def test_duplicate_breakpoints_handled(self):
        """Duplicate breakpoints should not create empty segments in the output."""
        from insurance_trend.breaks import split_segments
        t = np.arange(12, dtype=float)
        y = np.ones(12)
        segs = split_segments(t, y, [4, 4, 8])
        # After sort + dedup via sentinel: effectively [4, 4, 8, 12]
        # Duplicate 4 creates empty segment [4:4] which is filtered
        for s in segs:
            assert len(s[0]) > 0

    def test_t_and_y_have_same_length_in_each_segment(self):
        from insurance_trend.breaks import split_segments
        t = np.arange(20, dtype=float)
        y = np.arange(20, dtype=float) * 2
        segs = split_segments(t, y, [7, 14])
        for seg_t, seg_y in segs:
            assert len(seg_t) == len(seg_y)

    def test_values_preserved_in_segments(self):
        """y-values must be preserved exactly in the correct order."""
        from insurance_trend.breaks import split_segments
        t = np.arange(6, dtype=float)
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        segs = split_segments(t, y, [3])
        np.testing.assert_array_equal(segs[0][1], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(segs[1][1], [40.0, 50.0, 60.0])
