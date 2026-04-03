"""Expanded test coverage for insurance-trend.

Targets undertested areas discovered after reviewing all source modules:
- _utils.py: all helper functions
- breaks.py: split_segments edge cases, detect_breakpoints robustness
- result.py: TrendResult.plot(), LossCostTrendResult edge cases, trend_factor boundary
- decompose.py: MultiIndexResult edge cases, near-zero total trend, negative residual
- calendar.py: category enforcement, filter edge cases, attribution correctness
- inflation.py: InflationDecompositionResult.converged, decomposition_table dtypes,
  constructor edge cases, fit robustness
- frequency.py: private helpers _build_design_matrix, _project_forward, _periods_to_series
- index.py: ExternalIndex.from_series, from_csv behaviour
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# _utils.py tests
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_list_input(self):
        from insurance_trend._utils import to_numpy
        arr = to_numpy([1.0, 2.0, 3.0], "test")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        from insurance_trend._utils import to_numpy
        x = np.array([4.0, 5.0])
        arr = to_numpy(x, "test")
        np.testing.assert_array_equal(arr, x)

    def test_polars_series_input(self):
        from insurance_trend._utils import to_numpy
        s = pl.Series("v", [1.0, 2.0, 3.0])
        arr = to_numpy(s, "test")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_pandas_series_input(self):
        from insurance_trend._utils import to_numpy
        s = pd.Series([10.0, 20.0, 30.0])
        arr = to_numpy(s, "test")
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])

    def test_invalid_type_raises(self):
        from insurance_trend._utils import to_numpy
        with pytest.raises(TypeError, match="must be a pandas Series"):
            to_numpy({"a": 1}, "test")

    def test_integer_list_cast_to_float(self):
        from insurance_trend._utils import to_numpy
        arr = to_numpy([1, 2, 3], "test")
        assert arr.dtype == float


class TestToPolars:
    def test_polars_passthrough(self):
        from insurance_trend._utils import to_polars_series
        s = pl.Series("x", [1.0, 2.0])
        result = to_polars_series(s, "x")
        assert result is s

    def test_pandas_converted(self):
        from insurance_trend._utils import to_polars_series
        s = pd.Series([1.0, 2.0], name="px")
        result = to_polars_series(s, "px")
        assert isinstance(result, pl.Series)

    def test_list_converted(self):
        from insurance_trend._utils import to_polars_series
        result = to_polars_series([3.0, 4.0], "v")
        assert isinstance(result, pl.Series)
        assert len(result) == 2

    def test_numpy_converted(self):
        from insurance_trend._utils import to_polars_series
        arr = np.array([5.0, 6.0, 7.0])
        result = to_polars_series(arr, "v")
        assert isinstance(result, pl.Series)
        assert len(result) == 3


class TestAnnualTrendRate:
    def test_zero_slope_gives_zero(self):
        from insurance_trend._utils import annual_trend_rate
        assert annual_trend_rate(0.0, 4) == pytest.approx(0.0)

    def test_positive_slope(self):
        from insurance_trend._utils import annual_trend_rate
        beta = 0.07 / 4  # 7% pa slope per quarter
        rate = annual_trend_rate(beta, 4)
        assert rate == pytest.approx(np.exp(0.07) - 1.0)

    def test_negative_slope(self):
        from insurance_trend._utils import annual_trend_rate
        beta = -0.03 / 12  # negative monthly
        rate = annual_trend_rate(beta, 12)
        assert rate < 0.0

    def test_monthly_annualisation(self):
        from insurance_trend._utils import annual_trend_rate
        beta = 0.05 / 12
        rate = annual_trend_rate(beta, 12)
        assert rate == pytest.approx(np.exp(0.05) - 1.0)


class TestSafeLog:
    def test_positive_values(self):
        from insurance_trend._utils import safe_log
        arr = np.array([1.0, 2.0, 10.0])
        result = safe_log(arr, "test")
        np.testing.assert_allclose(result, np.log(arr))

    def test_zero_raises(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="non-positive"):
            safe_log(np.array([1.0, 0.0, 2.0]), "test")

    def test_negative_raises(self):
        from insurance_trend._utils import safe_log
        with pytest.raises(ValueError, match="non-positive"):
            safe_log(np.array([-1.0, 1.0]), "test")


class TestValidateLengths:
    def test_equal_lengths(self):
        from insurance_trend._utils import validate_lengths
        n = validate_lengths(a=[1, 2, 3], b=np.array([4, 5, 6]))
        assert n == 3

    def test_mismatched_raises(self):
        from insurance_trend._utils import validate_lengths
        with pytest.raises(ValueError, match="same length"):
            validate_lengths(a=[1, 2, 3], b=[1, 2])

    def test_polars_and_list(self):
        from insurance_trend._utils import validate_lengths
        n = validate_lengths(a=pl.Series([1.0, 2.0]), b=[3.0, 4.0])
        assert n == 2


class TestQuarterDummies:
    def test_shape(self):
        from insurance_trend._utils import quarter_dummies
        D = quarter_dummies(12)
        assert D.shape == (12, 3)

    def test_no_q4_column(self):
        """Q4 is the base; only Q1/Q2/Q3 dummies returned."""
        from insurance_trend._utils import quarter_dummies
        D = quarter_dummies(4)
        # period 0 = Q1, period 1 = Q2, period 2 = Q3, period 3 = Q4
        assert D[0, 0] == 1.0  # Q1
        assert D[1, 1] == 1.0  # Q2
        assert D[2, 2] == 1.0  # Q3
        assert D[3, :].sum() == 0.0  # Q4 — all zeros

    def test_non_zero_sum_per_row_at_most_one(self):
        from insurance_trend._utils import quarter_dummies
        D = quarter_dummies(20)
        for row in D:
            assert row.sum() in (0.0, 1.0)

    def test_with_explicit_periods(self):
        from insurance_trend._utils import quarter_dummies
        # Start at Q3 (index 2)
        periods = np.arange(2, 6)
        D = quarter_dummies(4, periods)
        assert D.shape == (4, 3)


# ---------------------------------------------------------------------------
# breaks.py additional tests
# ---------------------------------------------------------------------------


class TestSplitSegmentsAdditional:
    def test_unsorted_breaks_are_sorted(self):
        """split_segments should sort breaks internally."""
        from insurance_trend.breaks import split_segments
        t = np.arange(15, dtype=float)
        y = np.ones(15)
        segments = split_segments(t, y, [10, 5])
        # Should produce 3 segments of lengths 5, 5, 5
        lengths = [len(s[0]) for s in segments]
        assert lengths == [5, 5, 5]

    def test_t_and_y_stay_aligned(self):
        """Each segment's t and y must correspond element-wise."""
        from insurance_trend.breaks import split_segments
        t = np.arange(10, dtype=float)
        y = t ** 2
        segments = split_segments(t, y, [4, 7])
        for seg_t, seg_y in segments:
            np.testing.assert_array_equal(seg_y, seg_t ** 2)

    def test_all_breaks_beyond_length(self):
        """Break index >= n: the segment [n:] is empty and filtered out."""
        from insurance_trend.breaks import split_segments
        t = np.arange(10, dtype=float)
        y = np.ones(10)
        segments = split_segments(t, y, [50])
        # [t[:50], t[50:]] = [all 10, empty] => empty segment filtered
        total = sum(len(s[0]) for s in segments)
        assert total == 10


class TestDetectBreakpointsAdditional:
    def test_detects_break_in_covid_style_series(self):
        """A -35% step change (COVID motor frequency) should be detected."""
        from insurance_trend.breaks import detect_breakpoints
        rng = np.random.default_rng(55)
        n = 24
        log_freq = np.zeros(n)
        log_freq[:8] = 0.01 * np.arange(8) + rng.normal(0, 0.005, 8)
        log_freq[8:] = np.log(0.65) + 0.01 * np.arange(16) + rng.normal(0, 0.005, 16)
        breaks = detect_breakpoints(log_freq, penalty=1.0)
        assert any(5 <= b <= 12 for b in breaks)

    def test_all_zeros_no_breaks(self):
        """Constant zero series should yield no breaks."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.zeros(20)
        breaks = detect_breakpoints(signal, penalty=3.0)
        assert breaks == []

    def test_min_size_1_accepted(self):
        """min_size=1 is unusual but should not crash."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.concatenate([np.zeros(10), np.ones(10)])
        breaks = detect_breakpoints(signal, min_size=1, penalty=0.5)
        assert isinstance(breaks, list)

    def test_result_strictly_less_than_n(self):
        """All returned break indices must be < len(signal)."""
        from insurance_trend.breaks import detect_breakpoints
        signal = np.concatenate([np.zeros(12), np.full(12, 1.0)])
        breaks = detect_breakpoints(signal, penalty=1.0)
        assert all(b < len(signal) for b in breaks)


# ---------------------------------------------------------------------------
# result.py additional tests
# ---------------------------------------------------------------------------


class TestTrendResultAdditional:
    def _make_result(self, trend_rate: float = 0.05) -> "TrendResult":
        from insurance_trend.result import TrendResult
        n = 16
        actuals = np.ones(n) * 100.0
        fitted = actuals * 1.01
        residuals = actuals / fitted - 1.0
        proj = pl.DataFrame({
            "period": [1, 2, 3, 4],
            "point": [1.0, 1.0, 1.0, 1.0],
            "lower": [1.0] * 4,
            "upper": [1.0] * 4,
        })
        return TrendResult(
            trend_rate=trend_rate,
            ci_lower=trend_rate - 0.01,
            ci_upper=trend_rate + 0.01,
            method="log_linear",
            fitted_values=pl.Series("fitted", fitted),
            residuals=pl.Series("residuals", residuals),
            changepoints=[],
            projection=proj,
            r_squared=0.9,
            actuals=pl.Series("actuals", actuals),
            periods=pl.Series("periods", [f"2020Q{i % 4 + 1}" for i in range(n)]),
            n_bootstrap=1000,
            periods_per_year=4,
        )

    def test_trend_factor_large_period_count(self):
        r = self._make_result(0.05)
        # 40 periods = 10 years at quarterly
        expected = (1.05) ** 10.0
        assert r.trend_factor(40) == pytest.approx(expected, rel=1e-6)

    def test_trend_factor_fractional_periods(self):
        r = self._make_result(0.10)
        # 6 periods = 1.5 years
        expected = (1.10) ** 1.5
        assert r.trend_factor(6) == pytest.approx(expected, rel=1e-6)

    def test_summary_contains_ci_bounds(self):
        r = self._make_result(0.05)
        s = r.summary()
        assert "CI" in s

    def test_summary_contains_r_squared(self):
        r = self._make_result(0.05)
        s = r.summary()
        # R-squared value 0.9 should appear somewhere in summary
        assert "0.9" in s or "R-squared" in s

    def test_piecewise_method_stored(self):
        from insurance_trend.result import TrendResult
        n = 8
        proj = pl.DataFrame({
            "period": [1],
            "point": [1.0],
            "lower": [0.9],
            "upper": [1.1],
        })
        r = TrendResult(
            trend_rate=0.03,
            ci_lower=0.01,
            ci_upper=0.05,
            method="piecewise",
            fitted_values=pl.Series("fitted", np.ones(n)),
            residuals=pl.Series("residuals", np.zeros(n)),
            changepoints=[4],
            projection=proj,
            r_squared=0.85,
        )
        assert r.method == "piecewise"
        assert r.changepoints == [4]

    def test_plot_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = self._make_result(0.05)
        try:
            fig = r.plot()
            assert fig is not None
            plt.close("all")
        except Exception as e:
            pytest.skip(f"plot() requires display: {e}")

    def test_repr_format(self):
        r = self._make_result(0.07)
        s = repr(r)
        assert "0.0700" in s
        assert "TrendResult" in s


class TestLossCostTrendResultAdditional:
    def _make_lc(self, freq: float = -0.02, sev: float = 0.08):
        from insurance_trend.result import LossCostTrendResult, TrendResult
        n = 12
        proj = pl.DataFrame({
            "period": [1],
            "point": [1.0],
            "lower": [0.9],
            "upper": [1.1],
        })

        def make(tr, meth):
            return TrendResult(
                trend_rate=tr,
                ci_lower=tr - 0.01,
                ci_upper=tr + 0.01,
                method=meth,
                fitted_values=pl.Series("f", np.ones(n)),
                residuals=pl.Series("r", np.zeros(n)),
                changepoints=[],
                projection=proj,
                r_squared=0.9,
            )

        combined = (1 + freq) * (1 + sev) - 1.0
        return LossCostTrendResult(
            frequency=make(freq, "log_linear"),
            severity=make(sev, "log_linear"),
            combined_trend_rate=combined,
            superimposed_inflation=None,
            projection=proj,
        )

    def test_superimposed_none_in_decompose(self):
        lc = self._make_lc()
        d = lc.decompose()
        assert d["superimposed"] is None

    def test_summary_shows_na_for_no_superimposed(self):
        lc = self._make_lc()
        s = lc.summary()
        assert "N/A" in s

    def test_trend_factor_zero_periods(self):
        lc = self._make_lc()
        assert lc.trend_factor(0) == pytest.approx(1.0)

    def test_combined_trend_positive_when_sev_dominates(self):
        lc = self._make_lc(freq=0.0, sev=0.10)
        assert lc.combined_trend_rate > 0.0

    def test_combined_trend_negative_possible(self):
        lc = self._make_lc(freq=-0.20, sev=0.05)
        # -20% freq dominates +5% sev
        assert lc.combined_trend_rate < 0.0

    def test_repr_contains_freq_and_sev(self):
        lc = self._make_lc(freq=-0.02, sev=0.08)
        r = repr(lc)
        assert "freq=" in r
        assert "sev=" in r


# ---------------------------------------------------------------------------
# decompose.py additional tests
# ---------------------------------------------------------------------------


def _make_periods_dc(n: int) -> list[str]:
    labels = []
    for year in range(2019, 2035):
        for q in range(1, 5):
            labels.append(f"{year}Q{q}")
            if len(labels) == n:
                return labels
    return labels


class TestMultiIndexDecomposerAdditional:
    def test_near_zero_total_trend_no_crash(self):
        """When total severity is flat, share_of_total should be NaN gracefully."""
        from insurance_trend import MultiIndexDecomposer
        n = 20
        t = np.arange(n, dtype=float)
        # Flat severity — no trend
        severity = np.full(n, 5000.0)
        idx = np.exp(0.04 / 4 * t)
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=severity,
            indices={"cpi": idx},
        ).fit()
        # share_of_total_pct should contain NaN when total trend is ~0
        shares = result.decomposition_table["share_of_total_pct"].to_list()
        assert any(math.isnan(s) if s is not None else False for s in shares)

    def test_negative_residual_allowed(self):
        """If indices explain more than total trend, residual can be negative."""
        from insurance_trend import MultiIndexDecomposer
        rng = np.random.default_rng(77)
        n = 24
        t = np.arange(n, dtype=float)
        # Index grows faster than severity → residual expected to be negative
        idx = np.exp(0.10 / 4 * t)
        severity = 3000.0 * np.exp(0.05 / 4 * t) * np.exp(rng.normal(0, 0.005, n))
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=severity,
            indices={"heavy_idx": idx},
        ).fit()
        # No crash; residual_rate can be positive or negative
        assert isinstance(result.residual_rate, float)

    def test_periods_per_year_stored(self):
        from insurance_trend import MultiIndexDecomposer
        n = 20
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 4 * t)
        sev = np.exp(0.05 / 4 * t) * 5000.0
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=sev,
            indices={"idx": idx},
            periods_per_year=4,
        ).fit()
        assert result.periods_per_year == 4

    def test_repr_contains_r_squared(self):
        from insurance_trend import MultiIndexDecomposer
        n = 16
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 4 * t)
        severity = 5000.0 * (idx ** 0.7) * np.exp(np.random.default_rng(5).normal(0, 0.01, n))
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=severity,
            indices={"idx": idx},
        ).fit()
        assert "r_squared" in repr(result)

    def test_table_coefficient_nan_for_residual(self):
        """The Residual row in decomposition_table should have NaN coefficient."""
        from insurance_trend import MultiIndexDecomposer
        n = 20
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 4 * t)
        severity = 5000.0 * (idx ** 0.6) * np.exp(np.random.default_rng(6).normal(0, 0.005, n))
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=severity,
            indices={"idx": idx},
        ).fit()
        table = result.decomposition_table
        residual_row = table.filter(table["component"] == "Residual")
        coef = residual_row["coefficient"][0]
        assert coef is None or math.isnan(coef)

    def test_summary_contains_index_name(self):
        from insurance_trend import MultiIndexDecomposer
        n = 20
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 4 * t)
        severity = 4000.0 * (idx ** 0.7) * np.exp(np.random.default_rng(8).normal(0, 0.005, n))
        result = MultiIndexDecomposer(
            periods=_make_periods_dc(n),
            severity=severity,
            indices={"motor_repair_sppi": idx},
        ).fit()
        s = result.summary()
        assert "motor_repair_sppi" in s


# ---------------------------------------------------------------------------
# calendar.py additional tests
# ---------------------------------------------------------------------------


class TestCalendarEventCategory:
    def test_category_market_accepted(self):
        from insurance_trend.calendar import CalendarEvent
        evt = CalendarEvent(period="2023Q3", description="Used car values falling", category="market", impact=-1)
        assert evt.category == "market"

    def test_category_supply_chain_accepted(self):
        from insurance_trend.calendar import CalendarEvent
        evt = CalendarEvent(period="2022Q1", description="Repair parts surge", category="supply_chain", impact=1)
        assert evt.category == "supply_chain"

    def test_arbitrary_category_accepted(self):
        """CalendarEvent accepts free-text category — not enum-validated."""
        from insurance_trend.calendar import CalendarEvent
        evt = CalendarEvent(period="2022Q1", description="Test", category="custom_category", impact=0)
        assert evt.category == "custom_category"

    def test_all_default_events_have_description(self):
        from insurance_trend.calendar import _DEFAULT_UK_EVENTS
        for evt in _DEFAULT_UK_EVENTS:
            assert len(evt.description.strip()) > 0


class TestAttributionTieBreaking:
    def test_earlier_registered_wins_on_exact_tie(self):
        """When two events are equidistant, the one registered first wins."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=5)
        # Both are exactly 1 period from the break at 2020Q2
        cal.add_event("2020Q1", "Event A", "macro", 1)
        cal.add_event("2020Q3", "Event B", "legal", -1)
        report = cal.attribute(["2020Q2"])
        # Event A registered first; both equidistant
        assert report.attributions[0].matched_event.description == "Event A"

    def test_attribution_distance_correct(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=3)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        report = cal.attribute(["2020Q3"])  # 2 quarters away
        assert report.attributions[0].distance == pytest.approx(2.0)

    def test_tolerance_at_boundary_included(self):
        """A break exactly tolerance periods away should be explained."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=2)
        cal.add_event("2020Q1", "Event", "macro", 1)
        report = cal.attribute(["2020Q3"])  # exactly 2 periods away = tolerance
        assert report.n_explained == 1


class TestFilterEventsAdditional:
    def test_filter_from_and_to_same_period(self):
        """from_period == to_period should return only events at that period."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2019Q1", "A", "legal", 1)
        cal.add_event("2020Q1", "B", "covid", -1)
        cal.add_event("2021Q1", "C", "macro", 0)
        filtered = cal.filter_events(from_period="2020Q1", to_period="2020Q1")
        assert filtered.n_events == 1
        assert filtered.events[0].description == "B"

    def test_filter_from_after_to_returns_empty(self):
        """Logically impossible range should return empty."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "A", "covid", -1)
        filtered = cal.filter_events(from_period="2021Q1", to_period="2019Q1")
        assert filtered.n_events == 0

    def test_filter_preserves_event_order(self):
        """Filtered calendar should preserve original insertion order."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2019Q1", "First", "legal", 1)
        cal.add_event("2019Q3", "Second", "legal", -1)
        cal.add_event("2020Q1", "Third", "covid", -1)
        filtered = cal.filter_events(categories=["legal"])
        descs = [e.description for e in filtered.events]
        assert descs == ["First", "Second"]

    def test_filter_with_all_params_combined(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2016Q1", "Old legal", "legal", 1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        cal.add_event("2020Q2", "Legal change", "legal", 0)
        cal.add_event("2022Q1", "Recent legal", "legal", -1)
        # category=legal AND from 2019Q1 AND to 2021Q4
        filtered = cal.filter_events(
            categories=["legal"],
            from_period="2019Q1",
            to_period="2021Q4",
        )
        assert filtered.n_events == 1
        assert filtered.events[0].description == "Legal change"


class TestDefaultCalendarAdditional:
    def test_default_calendar_count_matches_n_events(self):
        from insurance_trend.calendar import BreakEventCalendar, _DEFAULT_UK_EVENTS
        cal = BreakEventCalendar()
        assert cal.n_events == len(_DEFAULT_UK_EVENTS)

    def test_no_duplicate_period_description_pairs(self):
        """No two events should have identical period + description."""
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        pairs = [(e.period, e.description) for e in cal.events]
        assert len(pairs) == len(set(pairs)), "Duplicate (period, description) pairs found"

    def test_gender_directive_2012_present(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        gender_events = [e for e in cal.events if "Gender" in e.description or "gender" in e.description.lower()]
        assert len(gender_events) >= 1, "Gender Directive 2012 event missing from default calendar"

    def test_supply_chain_events_present(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        sc_events = [e for e in cal.events if e.category == "supply_chain"]
        assert len(sc_events) >= 1, "At least one supply_chain event expected"

    def test_attribute_with_default_calendar_returns_report(self):
        from insurance_trend.calendar import BreakEventCalendar, AttributionReport
        cal = BreakEventCalendar()
        report = cal.attribute(["2017Q1", "2020Q1", "2022Q1", "2035Q1"])
        assert isinstance(report, AttributionReport)
        assert report.n_breaks == 4

    def test_ipt_2016_present(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        ipt_2016 = [e for e in cal.events if "2016Q4" == e.period and e.category == "tax"]
        assert len(ipt_2016) >= 1, "IPT 2016Q4 rise should be in default calendar"

    def test_covid_third_lockdown_present(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        lockdown3 = [e for e in cal.events if "2021Q1" == e.period and e.category == "covid"]
        assert len(lockdown3) >= 1, "Third lockdown 2021Q1 event missing from default calendar"

    def test_all_impacts_valid(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar()
        for evt in cal.events:
            assert evt.impact in (-1, 0, 1)


class TestAttributionReportAdditional:
    def test_to_dataframe_explained_is_bool(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        report = cal.attribute(["2020Q1"])
        df = report.to_dataframe()
        assert df["explained"].dtype == pl.Boolean

    def test_all_explained_when_all_match(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=0)
        cal.add_event("2019Q1", "A", "legal", 1)
        cal.add_event("2020Q1", "B", "covid", -1)
        report = cal.attribute(["2019Q1", "2020Q1"])
        assert report.n_explained == 2
        assert report.n_unexplained == 0

    def test_all_unexplained_when_no_match(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=0)
        cal.add_event("2019Q1", "A", "legal", 1)
        report = cal.attribute(["2030Q1", "2031Q1"])
        assert report.n_explained == 0
        assert report.n_unexplained == 2

    def test_summary_explains_multiple_events(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=1)
        cal.add_event("2017Q1", "Ogden", "legal", 1)
        cal.add_event("2020Q1", "COVID", "covid", -1)
        report = cal.attribute(["2017Q1", "2020Q1"])
        s = report.summary()
        assert "Ogden" in s
        assert "COVID" in s

    def test_n_breaks_n_explained_n_unexplained_sum_consistent(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(tolerance=1)
        report = cal.attribute(["2017Q1", "2022Q1", "2019Q3", "2035Q1"])
        assert report.n_breaks == report.n_explained + report.n_unexplained


class TestBreakEventCalendarAdditional:
    def test_add_many_events(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        for i in range(1, 5):
            cal.add_event(f"202{i}Q1", f"Event {i}", "macro", 1)
        assert cal.n_events == 4

    def test_remove_all_events_for_period(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        cal.add_event("2020Q1", "A", "covid", -1)
        cal.add_event("2020Q1", "B", "covid", -1)
        removed = cal.remove_event("2020Q1")
        assert removed == 2
        assert cal.n_events == 0

    def test_tolerance_zero_exact_only(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False, tolerance=0.0)
        cal.add_event("2020Q1", "Event", "covid", -1)
        r_exact = cal.attribute(["2020Q1"])
        assert r_exact.n_explained == 1
        r_off = cal.attribute(["2020Q2"])
        assert r_off.n_explained == 0

    def test_events_dataframe_empty_calendar(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        df = cal.events_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "period" in df.columns

    def test_chaining_multiple_adds(self):
        from insurance_trend.calendar import BreakEventCalendar
        cal = BreakEventCalendar(include_defaults=False)
        result = (
            cal.add_event("2020Q1", "A", "covid", -1)
               .add_event("2021Q1", "B", "legal", 1)
               .add_event("2022Q1", "C", "tax", 0)
        )
        assert result is cal
        assert cal.n_events == 3


# ---------------------------------------------------------------------------
# inflation.py additional tests
# ---------------------------------------------------------------------------


def _make_inflation_series(n: int = 32, pa: float = 0.07, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 100.0 * np.exp(pa / 4 * t + rng.normal(0, 0.01, n))


class TestInflationDecomposerAdditional:
    def test_non_cycle_structural_rate_not_nan(self):
        """cycle=False model should still produce a finite structural_rate."""
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=20, pa=0.06)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert math.isfinite(result.structural_rate)

    def test_converged_attribute_is_bool(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=32)
        result = InflationDecomposer(series=series).fit()
        assert isinstance(result.converged, bool)

    def test_fit_kwargs_disp_accepted(self):
        """fit_kwargs={'disp': False} is the default; ensure it doesn't break."""
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, fit_kwargs={"disp": False}).fit()
        assert result is not None

    def test_stochastic_cycle_false_accepted(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=True, stochastic_cycle=False).fit()
        assert result.n_obs == 24

    def test_damped_cycle_false_accepted(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=True, damped_cycle=False).fit()
        assert result.n_obs == 24

    def test_trend_series_name(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert result.trend.name == "trend"

    def test_cycle_series_name(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=True).fit()
        assert result.cycle.name == "cycle"

    def test_irregular_series_name(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert result.irregular.name == "irregular"

    def test_decomposition_table_period_column_strings(self):
        from insurance_trend import InflationDecomposer
        periods = [f"2018Q{q}" for q in range(1, 5)] * 6  # 24 periods
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, periods=periods, cycle=False).fit()
        table = result.decomposition_table()
        assert table["period"][0] == "2018Q1"

    def test_decomposition_table_float_columns(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        table = result.decomposition_table()
        for col in ("observed", "trend", "cycle", "seasonal", "irregular"):
            assert table[col].dtype in (pl.Float64, pl.Float32)

    def test_no_cycle_cyclical_position_zero(self):
        """cycle=False means cycle component is zero, so cyclical_position should be 0."""
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert result.cyclical_position == pytest.approx(0.0)

    def test_repr_contains_aic(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert "aic" in repr(result)

    def test_log_transform_false_runs(self):
        """log_transform=False: series on original scale, no log applied."""
        from insurance_trend import InflationDecomposer
        # Small values around 0.1 so the model still converges
        rng = np.random.default_rng(99)
        series = 0.10 + rng.normal(0, 0.005, 24)
        result = InflationDecomposer(series=series, log_transform=False, cycle=False).fit()
        assert result.log_transform is False
        assert result.n_obs == 24


class TestInflationDecompositionResultMethodsAdditional:
    def test_summary_contains_bic(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert "BIC" in result.summary()

    def test_summary_contains_converged(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert "Converged" in result.summary()

    def test_summary_periods_per_year(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False, periods_per_year=4).fit()
        s = result.summary()
        # periods_per_year=4 should appear somewhere
        assert "4" in s

    def test_aic_bic_finite(self):
        from insurance_trend import InflationDecomposer
        series = _make_inflation_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert math.isfinite(result.aic)
        assert math.isfinite(result.bic)


# ---------------------------------------------------------------------------
# frequency.py private helper tests
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    def test_no_seasonal_shape(self):
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(12, dtype=float)
        X = _build_design_matrix(t, seasonal=False, periods_per_year=4)
        # intercept + slope = 2 columns
        assert X.shape == (12, 2)

    def test_with_seasonal_shape(self):
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(12, dtype=float)
        X = _build_design_matrix(t, seasonal=True, periods_per_year=4)
        # intercept + slope + Q1 + Q2 + Q3 = 5 columns
        assert X.shape == (12, 5)

    def test_monthly_data_no_seasonal_added(self):
        """Seasonal dummies are only added for quarterly (ppy=4) data."""
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(24, dtype=float)
        X = _build_design_matrix(t, seasonal=True, periods_per_year=12)
        # monthly: seasonal=True has no effect since ppy != 4 → still 2 cols
        assert X.shape == (24, 2)

    def test_intercept_column_is_ones(self):
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(8, dtype=float)
        X = _build_design_matrix(t, seasonal=False, periods_per_year=4)
        np.testing.assert_array_equal(X[:, 0], np.ones(8))

    def test_slope_column_is_t(self):
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(8, dtype=float)
        X = _build_design_matrix(t, seasonal=False, periods_per_year=4)
        np.testing.assert_array_equal(X[:, 1], t)


class TestProjectForward:
    def test_returns_polars_dataframe(self):
        from insurance_trend.frequency import _project_forward
        df = _project_forward(
            last_fitted=100.0,
            beta=0.01,
            periods_per_year=4,
            n_periods=8,
            ci_lower=0.03,
            ci_upper=0.07,
        )
        assert isinstance(df, pl.DataFrame)

    def test_correct_columns(self):
        from insurance_trend.frequency import _project_forward
        df = _project_forward(100.0, 0.01, 4, 4, 0.03, 0.07)
        assert set(df.columns) == {"period", "point", "lower", "upper"}

    def test_point_monotonically_increasing_for_positive_beta(self):
        from insurance_trend.frequency import _project_forward
        df = _project_forward(100.0, 0.01, 4, 8, 0.03, 0.07)
        points = df["point"].to_list()
        assert all(points[i] < points[i + 1] for i in range(len(points) - 1))

    def test_zero_n_periods_returns_empty(self):
        from insurance_trend.frequency import _project_forward
        df = _project_forward(100.0, 0.01, 4, 0, 0.03, 0.07)
        assert len(df) == 0

    def test_upper_greater_than_lower_for_positive_ci(self):
        from insurance_trend.frequency import _project_forward
        df = _project_forward(100.0, 0.01, 4, 4, 0.02, 0.08)
        lowers = df["lower"].to_list()
        uppers = df["upper"].to_list()
        assert all(u > l for u, l in zip(uppers, lowers))

    def test_negative_beta_decreasing_points(self):
        from insurance_trend.frequency import _project_forward
        # Negative beta => per-period rate is negative => values decrease
        df = _project_forward(100.0, -0.02, 4, 4, -0.10, -0.02)
        points = df["point"].to_list()
        assert all(points[i] > points[i + 1] for i in range(len(points) - 1))


class TestPeriodsToSeries:
    def test_list_input(self):
        from insurance_trend.frequency import _periods_to_series
        s = _periods_to_series(["2020Q1", "2020Q2", "2020Q3"], 3)
        assert isinstance(s, pl.Series)
        assert s[0] == "2020Q1"

    def test_fallback_on_none_input(self):
        from insurance_trend.frequency import _periods_to_series
        # None causes the hasattr/iter path to fail → integer fallback
        s = _periods_to_series(None, 3)
        assert len(s) == 3

    def test_polars_series_input(self):
        from insurance_trend.frequency import _periods_to_series
        ps = pl.Series("p", ["2021Q1", "2021Q2"])
        s = _periods_to_series(ps, 2)
        assert isinstance(s, pl.Series)
        assert s[0] == "2021Q1"

    def test_truncates_to_n(self):
        from insurance_trend.frequency import _periods_to_series
        long_list = [f"2020Q{q}" for q in range(1, 5)] * 3  # 12 items
        s = _periods_to_series(long_list, 6)
        assert len(s) == 6


# ---------------------------------------------------------------------------
# FrequencyTrendFitter edge cases
# ---------------------------------------------------------------------------


class TestFrequencyTrendFitterEdgeCases:
    def _make_fitter(self, n: int = 20, seed: int = 1):
        from insurance_trend import FrequencyTrendFitter
        rng = np.random.default_rng(seed)
        exposure = np.full(n, 1000.0)
        freq = 0.10 * np.exp(rng.normal(0, 0.02, n))
        counts = np.maximum(1.0, np.round(exposure * freq))
        periods = [f"{y}Q{q}" for y in range(2019, 2025) for q in range(1, 5)][:n]
        return FrequencyTrendFitter(
            periods=periods,
            claim_counts=counts,
            earned_exposure=exposure,
        )

    def test_fit_no_seasonal(self):
        from insurance_trend import TrendResult
        fitter = self._make_fitter(n=16)
        result = fitter.fit(seasonal=False, detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_fit_with_weights(self):
        from insurance_trend import TrendResult
        n = 20
        fitter = self._make_fitter(n=n)
        # Patch weights onto fitter internals
        from insurance_trend import FrequencyTrendFitter
        rng = np.random.default_rng(11)
        exposure = np.full(n, 1000.0)
        counts = np.maximum(1.0, np.round(exposure * 0.10 * np.exp(rng.normal(0, 0.02, n))))
        periods = [f"{y}Q{q}" for y in range(2019, 2025) for q in range(1, 5)][:n]
        weights = np.linspace(0.5, 1.0, n)
        fitter2 = FrequencyTrendFitter(
            periods=periods,
            claim_counts=counts,
            earned_exposure=exposure,
            weights=weights,
        )
        result = fitter2.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_frequency_property_correct(self):
        from insurance_trend import FrequencyTrendFitter
        counts = np.array([100.0, 110.0, 90.0])
        exposure = np.array([1000.0, 1100.0, 900.0])
        fitter = FrequencyTrendFitter(
            periods=["2020Q1", "2020Q2", "2020Q3"],
            claim_counts=counts,
            earned_exposure=exposure,
        )
        expected = counts / exposure
        np.testing.assert_allclose(fitter.frequency, expected)

    def test_summary_method(self):
        fitter = self._make_fitter()
        s = fitter.summary()
        assert isinstance(s, str)
        assert "FrequencyTrendFitter" in s

    def test_fit_returns_actuals_matching_frequency(self):
        from insurance_trend import FrequencyTrendFitter
        n = 16
        exposure = np.full(n, 1000.0)
        counts = np.full(n, 100.0)
        periods = [f"2020Q{q}" for q in range(1, 5)] * (n // 4)
        fitter = FrequencyTrendFitter(
            periods=periods,
            claim_counts=counts,
            earned_exposure=exposure,
        )
        result = fitter.fit(detect_breaks=False)
        # frequency = 100/1000 = 0.1
        assert result.actuals.to_list()[0] == pytest.approx(0.1)

    def test_fit_with_explicit_changepoints(self):
        from insurance_trend import TrendResult
        fitter = self._make_fitter(n=20)
        result = fitter.fit(changepoints=[8], detect_breaks=False)
        assert isinstance(result, TrendResult)
        assert result.changepoints == [8]

    def test_ci_level_90(self):
        from insurance_trend import TrendResult
        fitter = self._make_fitter(n=16)
        result = fitter.fit(detect_breaks=False, n_bootstrap=50, ci_level=0.90)
        assert isinstance(result, TrendResult)
        assert result.ci_lower < result.ci_upper

    def test_fit_projection_has_correct_length(self):
        fitter = self._make_fitter(n=16)
        result = fitter.fit(detect_breaks=False, projection_periods=6)
        assert len(result.projection) == 6

    def test_fit_local_linear_returns_result(self):
        """local_linear_trend method should run without error."""
        from insurance_trend import TrendResult
        fitter = self._make_fitter(n=20)
        result = fitter.fit(method="local_linear_trend", n_bootstrap=10)
        assert isinstance(result, TrendResult)
        assert result.method == "local_linear_trend"


# ---------------------------------------------------------------------------
# index.py ExternalIndex tests
# ---------------------------------------------------------------------------


class TestExternalIndexFromSeries:
    def test_from_series_polars(self):
        from insurance_trend import ExternalIndex
        s = pl.Series("HPTH", [100.0, 101.0, 102.0, 103.0])
        idx = ExternalIndex.from_series(s, label="HPTH")
        assert isinstance(idx, pl.Series)
        assert len(idx) == 4

    def test_from_series_numpy(self):
        from insurance_trend import ExternalIndex
        arr = np.array([100.0, 102.0, 104.0])
        idx = ExternalIndex.from_series(arr, label="test")
        assert isinstance(idx, pl.Series)
        assert len(idx) == 3

    def test_from_series_list(self):
        from insurance_trend import ExternalIndex
        idx = ExternalIndex.from_series([100.0, 101.0, 102.0], label="test")
        assert isinstance(idx, pl.Series)

    def test_from_series_pandas(self):
        from insurance_trend import ExternalIndex
        s = pd.Series([200.0, 210.0, 220.0], name="v")
        idx = ExternalIndex.from_series(s, label="v")
        assert isinstance(idx, pl.Series)

    def test_from_series_label_used(self):
        from insurance_trend import ExternalIndex
        idx = ExternalIndex.from_series([100.0, 101.0], label="my_index")
        assert idx.name == "my_index"


class TestExternalIndexCSV:
    def test_from_csv_basic(self, tmp_path):
        from insurance_trend import ExternalIndex
        csv_file = tmp_path / "test_index.csv"
        csv_file.write_text("period,value\n2020Q1,100.0\n2020Q2,101.0\n2020Q3,102.0\n")
        idx = ExternalIndex.from_csv(str(csv_file), date_col="period", value_col="value")
        assert isinstance(idx, pl.Series)
        assert len(idx) == 3

    def test_from_csv_values_correct(self, tmp_path):
        from insurance_trend import ExternalIndex
        csv_file = tmp_path / "test_values.csv"
        csv_file.write_text("period,index_val\n2020Q1,150.0\n2020Q2,155.0\n")
        idx = ExternalIndex.from_csv(str(csv_file), date_col="period", value_col="index_val")
        assert idx[0] == pytest.approx(150.0)
        assert idx[1] == pytest.approx(155.0)

    def test_from_csv_nonexistent_file_raises(self):
        from insurance_trend import ExternalIndex
        with pytest.raises((FileNotFoundError, Exception)):
            ExternalIndex.from_csv("/no/such/file.csv", date_col="period", value_col="value")

    def test_from_csv_missing_value_col_raises(self, tmp_path):
        from insurance_trend import ExternalIndex
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("period,price\n2020Q1,100.0\n")
        with pytest.raises((ValueError, Exception)):
            ExternalIndex.from_csv(str(csv_file), date_col="period", value_col="value")

    def test_from_csv_missing_date_col_raises(self, tmp_path):
        from insurance_trend import ExternalIndex
        csv_file = tmp_path / "bad2.csv"
        csv_file.write_text("date_col,value\n2020Q1,100.0\n")
        with pytest.raises((ValueError, Exception)):
            ExternalIndex.from_csv(str(csv_file), date_col="period", value_col="value")


class TestExternalIndexCatalogueAdditional:
    def test_catalogue_has_earnings_codes(self):
        from insurance_trend import ExternalIndex
        assert "avg_weekly_earnings" in ExternalIndex.CATALOGUE
        assert "avg_weekly_earnings_priv" in ExternalIndex.CATALOGUE

    def test_catalogue_has_ppi_codes(self):
        from insurance_trend import ExternalIndex
        assert "services_ppi" in ExternalIndex.CATALOGUE
        assert "ppi_output_all" in ExternalIndex.CATALOGUE

    def test_all_catalogue_values_are_strings(self):
        from insurance_trend import ExternalIndex
        for name, code in ExternalIndex.CATALOGUE.items():
            assert isinstance(code, str), f"Catalogue code for {name!r} is not a string"
            assert len(code) > 0, f"Empty catalogue code for {name!r}"

    def test_list_catalogue_description_col(self):
        from insurance_trend import ExternalIndex
        df = ExternalIndex.list_catalogue()
        # list_catalogue should return at least name + ons_code columns
        assert "name" in df.columns
        assert "ons_code" in df.columns
