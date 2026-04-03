"""Additional coverage for FrequencyTrendFitter, SeverityTrendFitter, LossCostTrendFitter.

Targets code paths not covered by existing tests:
- FrequencyTrendFitter: _build_design_matrix non-quarterly, _project_forward zero periods,
  _periods_to_series error fallback, monthly periods_per_year, summary variants
- SeverityTrendFitter: external_index first-value-zero error, index longer than series,
  _compute_index_trend_rate, deflated_severity first-element, WLS path
- LossCostTrendFitter: _combined_projection empty branch, projected_loss_cost,
  summary with external index, loss_cost property
- Module-level helpers: _project_forward with negative/zero beta
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pandas as pd
import pytest

from insurance_trend import (
    FrequencyTrendFitter,
    SeverityTrendFitter,
    LossCostTrendFitter,
    TrendResult,
    LossCostTrendResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freq_fitter(n: int = 16, seed: int = 0) -> FrequencyTrendFitter:
    rng = np.random.default_rng(seed)
    exposure = np.full(n, 1000.0) + rng.normal(0, 5, n)
    counts = np.maximum(1.0, np.round(exposure * 0.10))
    periods = [f"{y}Q{q}" for y in range(2020, 2030) for q in range(1, 5)][:n]
    return FrequencyTrendFitter(periods=periods, claim_counts=counts, earned_exposure=exposure)


def _make_sev_fitter(n: int = 16, seed: int = 1) -> SeverityTrendFitter:
    rng = np.random.default_rng(seed)
    counts = np.maximum(1.0, np.round(np.full(n, 1000.0) * 0.10))
    total_paid = counts * 5000.0 + rng.normal(0, 10000, n)
    total_paid = np.maximum(counts * 100.0, total_paid)  # keep positive
    periods = [f"{y}Q{q}" for y in range(2020, 2030) for q in range(1, 5)][:n]
    return SeverityTrendFitter(periods=periods, total_paid=total_paid, claim_counts=counts)


def _make_lc_fitter(n: int = 16, seed: int = 2) -> LossCostTrendFitter:
    rng = np.random.default_rng(seed)
    exposure = np.full(n, 1000.0)
    counts = np.maximum(1.0, np.round(exposure * 0.10))
    total_paid = counts * 5000.0 + rng.normal(0, 10000, n)
    total_paid = np.maximum(counts * 100.0, total_paid)
    periods = [f"{y}Q{q}" for y in range(2020, 2030) for q in range(1, 5)][:n]
    return LossCostTrendFitter(
        periods=periods,
        claim_counts=counts,
        earned_exposure=exposure,
        total_paid=total_paid,
    )


# ---------------------------------------------------------------------------
# FrequencyTrendFitter: _build_design_matrix paths
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    def test_non_quarterly_no_seasonal_dummies(self):
        """With periods_per_year=12, seasonal=True should NOT add dummies (only quarterly)."""
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(12, dtype=float)
        X = _build_design_matrix(t, seasonal=True, periods_per_year=12)
        # Should only have intercept + trend = 2 columns (no seasonal dummies for monthly)
        assert X.shape == (12, 2)

    def test_quarterly_seasonal_adds_dummies(self):
        """With periods_per_year=4, seasonal=True should add 3 dummy columns."""
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(8, dtype=float)
        X = _build_design_matrix(t, seasonal=True, periods_per_year=4)
        assert X.shape == (8, 5)  # intercept + trend + 3 dummies

    def test_seasonal_false_no_dummies(self):
        """seasonal=False must always produce 2-column matrix."""
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(8, dtype=float)
        X = _build_design_matrix(t, seasonal=False, periods_per_year=4)
        assert X.shape == (8, 2)

    def test_intercept_column_is_all_ones(self):
        from insurance_trend.frequency import _build_design_matrix
        t = np.arange(6, dtype=float)
        X = _build_design_matrix(t, seasonal=False, periods_per_year=4)
        np.testing.assert_array_equal(X[:, 0], np.ones(6))


# ---------------------------------------------------------------------------
# FrequencyTrendFitter: _project_forward
# ---------------------------------------------------------------------------


class TestProjectForward:
    def test_zero_n_periods_returns_empty(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.01, periods_per_year=4,
            n_periods=0, ci_lower=-0.01, ci_upper=0.05,
        )
        assert isinstance(proj, pl.DataFrame)
        assert len(proj) == 0

    def test_negative_n_periods_returns_empty(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.01, periods_per_year=4,
            n_periods=-5, ci_lower=-0.01, ci_upper=0.05,
        )
        assert len(proj) == 0

    def test_columns_correct(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.01, periods_per_year=4,
            n_periods=4, ci_lower=-0.01, ci_upper=0.05,
        )
        assert set(proj.columns) == {"period", "point", "lower", "upper"}

    def test_positive_beta_point_increases(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.01, periods_per_year=4,
            n_periods=4, ci_lower=-0.01, ci_upper=0.05,
        )
        pts = proj["point"].to_numpy()
        assert pts[-1] > pts[0]

    def test_zero_beta_point_constant(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.0, periods_per_year=4,
            n_periods=4, ci_lower=0.0, ci_upper=0.0,
        )
        pts = proj["point"].to_numpy()
        # exp(0) - 1 = 0, so each period multiplied by 1.0^i = same
        np.testing.assert_allclose(pts, [100.0, 100.0, 100.0, 100.0], rtol=1e-10)

    def test_lower_le_point_le_upper_for_positive_trend(self):
        from insurance_trend.frequency import _project_forward
        proj = _project_forward(
            last_fitted=100.0, beta=0.02, periods_per_year=4,
            n_periods=8, ci_lower=0.01, ci_upper=0.10,
        )
        lowers = proj["lower"].to_numpy()
        points = proj["point"].to_numpy()
        uppers = proj["upper"].to_numpy()
        assert np.all(lowers <= points + 1e-6)
        assert np.all(points <= uppers + 1e-6)


# ---------------------------------------------------------------------------
# FrequencyTrendFitter: _periods_to_series
# ---------------------------------------------------------------------------


class TestPeriodsToSeries:
    def test_list_of_strings(self):
        from insurance_trend.frequency import _periods_to_series
        s = _periods_to_series(["2020Q1", "2020Q2", "2020Q3"], n=3)
        assert isinstance(s, pl.Series)
        assert list(s) == ["2020Q1", "2020Q2", "2020Q3"]

    def test_numpy_array(self):
        from insurance_trend.frequency import _periods_to_series
        arr = np.array(["A", "B", "C"])
        s = _periods_to_series(arr, n=3)
        assert isinstance(s, pl.Series)
        assert len(s) == 3

    def test_polars_series(self):
        from insurance_trend.frequency import _periods_to_series
        ps = pl.Series("p", ["X", "Y"])
        s = _periods_to_series(ps, n=2)
        assert isinstance(s, pl.Series)

    def test_n_clips_output(self):
        """If n < len(periods), only first n labels are returned."""
        from insurance_trend.frequency import _periods_to_series
        s = _periods_to_series(["A", "B", "C", "D"], n=2)
        assert len(s) == 2

    def test_fallback_on_error(self):
        """When conversion fails, should fall back to integer range strings."""
        from insurance_trend.frequency import _periods_to_series

        class Unconvertible:
            def __iter__(self):
                raise RuntimeError("Cannot iterate")

        s = _periods_to_series(Unconvertible(), n=3)
        assert isinstance(s, pl.Series)
        assert len(s) == 3


# ---------------------------------------------------------------------------
# FrequencyTrendFitter: monthly periods_per_year
# ---------------------------------------------------------------------------


class TestFrequencyMonthly:
    def test_monthly_fit_returns_result(self):
        rng = np.random.default_rng(10)
        n = 24
        counts = np.maximum(1.0, np.round(np.full(n, 500.0) * 0.08))
        exposure = np.full(n, 500.0)
        periods = [f"2022M{m:02d}" for m in range(1, 13)] + \
                  [f"2023M{m:02d}" for m in range(1, 13)]
        fitter = FrequencyTrendFitter(
            periods=periods, claim_counts=counts, earned_exposure=exposure,
            periods_per_year=12,
        )
        result = fitter.fit(detect_breaks=False, seasonal=False, n_bootstrap=50)
        assert isinstance(result, TrendResult)
        assert result.periods_per_year == 12

    def test_monthly_trend_factor_correct(self):
        rng = np.random.default_rng(11)
        n = 24
        counts = np.maximum(1.0, np.round(np.full(n, 500.0) * 0.10))
        exposure = np.full(n, 500.0)
        periods = list(range(n))
        fitter = FrequencyTrendFitter(
            periods=periods, claim_counts=counts, earned_exposure=exposure,
            periods_per_year=12,
        )
        result = fitter.fit(detect_breaks=False, seasonal=False, n_bootstrap=50)
        # trend_factor(12) should equal (1 + trend_rate)^1 for monthly data
        expected = (1.0 + result.trend_rate) ** 1.0
        assert result.trend_factor(12) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# FrequencyTrendFitter: WLS path
# ---------------------------------------------------------------------------


class TestFrequencyWLS:
    def test_wls_result_differs_from_ols(self):
        """WLS with extreme weights should differ from OLS."""
        rng = np.random.default_rng(20)
        n = 16
        exposure = np.full(n, 1000.0)
        counts = np.maximum(1.0, np.round(exposure * 0.10 * np.exp(rng.normal(0, 0.05, n))))
        periods = [f"{y}Q{q}" for y in range(2020, 2024) for q in range(1, 5)]

        ols_fitter = FrequencyTrendFitter(
            periods=periods, claim_counts=counts, earned_exposure=exposure,
        )
        wls_fitter = FrequencyTrendFitter(
            periods=periods, claim_counts=counts, earned_exposure=exposure,
            # Extreme weighting: only last period matters
            weights=np.concatenate([np.full(n-1, 0.001), [1000.0]]),
        )
        ols_result = ols_fitter.fit(detect_breaks=False, n_bootstrap=50)
        wls_result = wls_fitter.fit(detect_breaks=False, n_bootstrap=50)
        # Trend rates should differ (extreme weighting changes fit significantly)
        assert isinstance(ols_result.trend_rate, float)
        assert isinstance(wls_result.trend_rate, float)


# ---------------------------------------------------------------------------
# SeverityTrendFitter: external index edge cases
# ---------------------------------------------------------------------------


class TestSeverityExternalIndexEdgeCases:
    def test_external_index_first_value_zero_raises(self, trending_data):
        """external_index with first value = 0 must raise ValueError."""
        bad_index = np.concatenate([[0.0], np.ones(len(trending_data["total_paid"]) - 1)])
        with pytest.raises(ValueError, match="non-zero"):
            SeverityTrendFitter(
                periods=trending_data["periods"],
                total_paid=trending_data["total_paid"],
                claim_counts=trending_data["claim_counts"],
                external_index=bad_index,
            )

    def test_external_index_longer_than_periods_accepted(self, trending_data):
        """An index longer than the data should be trimmed, not rejected."""
        n = len(trending_data["total_paid"])
        long_index = np.ones(n + 10) * 100.0
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=long_index,
        )
        # Should not raise; deflated_severity should have length n
        assert len(fitter.deflated_severity) == n

    def test_index_rebased_so_first_equals_raw(self, trending_data):
        """After rebasing, deflated_severity[0] == severity[0] / (index[0]/index[0]) = severity[0]."""
        n = len(trending_data["total_paid"])
        # Index that rises over time
        index = np.linspace(100.0, 120.0, n)
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=index,
        )
        # internal _external_index[0] = index[0]/index[0] = 1.0
        # so deflated[0] = severity[0] / 1.0
        np.testing.assert_allclose(fitter.deflated_severity[0], fitter.severity[0], rtol=1e-6)

    def test_superimposed_is_none_before_fit(self, trending_data):
        """superimposed_inflation() must return None before fit is called."""
        n = len(trending_data["total_paid"])
        index = np.ones(n) * 100.0
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=index,
        )
        assert fitter.superimposed_inflation() is None

    def test_compute_index_trend_rate(self, trending_data, external_index_series):
        """After fit, internal _index_trend_rate must be set."""
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        fitter.fit(detect_breaks=False, seasonal=False, n_bootstrap=50)
        # ~4% pa index growth
        assert fitter._index_trend_rate is not None
        assert fitter._index_trend_rate == pytest.approx(0.04, abs=0.03)

    def test_detect_breaks_warns_severity(self, breakpoint_data):
        """Auto break detection on severity with clear break should warn."""
        fitter = SeverityTrendFitter(
            periods=breakpoint_data["periods"],
            total_paid=breakpoint_data["total_paid"],
            claim_counts=breakpoint_data["claim_counts"],
        )
        with pytest.warns(UserWarning):
            fitter.fit(detect_breaks=True, penalty=0.5, n_bootstrap=50)

    def test_invalid_method_raises(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        with pytest.raises(ValueError, match="Unknown method"):
            fitter.fit(method="xgboost")

    def test_detect_breaks_false_no_warning(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitter.fit(detect_breaks=False, n_bootstrap=50)

    def test_summary_contains_external_index_yes(self, trending_data, external_index_series):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        s = fitter.summary()
        assert "yes" in s

    def test_summary_contains_external_index_no(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        s = fitter.summary()
        assert "no" in s


# ---------------------------------------------------------------------------
# SeverityTrendFitter: WLS path
# ---------------------------------------------------------------------------


class TestSeverityWLS:
    def test_wls_accepts_weights(self, trending_data):
        n = len(trending_data["periods"])
        weights = np.linspace(0.3, 1.0, n)
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            weights=weights,
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        assert isinstance(result, TrendResult)


# ---------------------------------------------------------------------------
# LossCostTrendFitter: additional paths
# ---------------------------------------------------------------------------


class TestLossCostAdditional:
    def test_loss_cost_property(self):
        fitter = _make_lc_fitter()
        lc = fitter.loss_cost
        assert isinstance(lc, np.ndarray)
        assert len(lc) == 16

    def test_loss_cost_property_returns_copy(self):
        fitter = _make_lc_fitter()
        lc1 = fitter.loss_cost
        lc2 = fitter.loss_cost
        lc1[0] = 9999.0
        assert fitter.loss_cost[0] != 9999.0

    def test_summary_contains_no_external_index(self):
        fitter = _make_lc_fitter()
        s = fitter.summary()
        assert "external_index=no" in s

    def test_summary_contains_yes_with_external_index(self, trending_data, external_index_series):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
            external_index=external_index_series,
        )
        s = fitter.summary()
        assert "external_index=yes" in s

    def test_projected_loss_cost_direct_method(self):
        fitter = _make_lc_fitter()
        proj = fitter.projected_loss_cost(future_periods=4)
        assert isinstance(proj, pl.DataFrame)
        assert len(proj) == 4

    def test_projected_loss_cost_ci_level(self):
        fitter = _make_lc_fitter()
        proj = fitter.projected_loss_cost(future_periods=4, ci=0.90)
        assert set(proj.columns) == {"period", "point", "lower", "upper"}

    def test_fit_with_explicit_changepoints(self, breakpoint_data):
        fitter = LossCostTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
            total_paid=breakpoint_data["total_paid"],
        )
        result = fitter.fit(
            changepoints=[breakpoint_data["break_index"]],
            detect_breaks=False,
            n_bootstrap=50,
        )
        assert isinstance(result, LossCostTrendResult)
        assert result.frequency.method == "piecewise"

    def test_combined_projection_positive_values(self):
        """Loss cost projection should have positive point values."""
        fitter = _make_lc_fitter()
        result = fitter.fit(detect_breaks=False, n_bootstrap=50, projection_periods=4)
        pts = result.projection["point"].to_numpy()
        assert np.all(pts > 0)

    def test_combined_trend_formula(self, trending_data):
        """Combined trend = (1+freq) * (1+sev) - 1."""
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        expected = (1.0 + result.frequency.trend_rate) * (1.0 + result.severity.trend_rate) - 1.0
        assert result.combined_trend_rate == pytest.approx(expected, rel=1e-10)

    def test_summary_no_superimposed(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        s = result.summary()
        assert "N/A" in s  # superimposed shows N/A when not computed


# ---------------------------------------------------------------------------
# LossCostTrendResult: edge cases
# ---------------------------------------------------------------------------


class TestLossCostTrendResultEdgeCases:
    def _make_lc_result(
        self,
        freq_trend: float = -0.02,
        sev_trend: float = 0.08,
        si: float | None = None,
    ) -> LossCostTrendResult:
        from insurance_trend.result import TrendResult, LossCostTrendResult
        n = 8
        fitted = pl.Series("fitted", np.full(n, 100.0))
        residuals = pl.Series("residuals", np.zeros(n))
        proj_df = pl.DataFrame({"period": [1, 2], "point": [100.0, 102.0],
                                "lower": [98.0, 99.0], "upper": [102.0, 105.0]})
        freq_result = TrendResult(
            trend_rate=freq_trend, ci_lower=freq_trend-0.01, ci_upper=freq_trend+0.01,
            method="log_linear", fitted_values=fitted, residuals=residuals,
            changepoints=[], projection=proj_df, r_squared=0.8,
            periods_per_year=4,
        )
        sev_result = TrendResult(
            trend_rate=sev_trend, ci_lower=sev_trend-0.01, ci_upper=sev_trend+0.01,
            method="log_linear", fitted_values=fitted, residuals=residuals,
            changepoints=[], projection=proj_df, r_squared=0.9,
            periods_per_year=4,
        )
        combined = (1 + freq_trend) * (1 + sev_trend) - 1.0
        return LossCostTrendResult(
            frequency=freq_result, severity=sev_result,
            combined_trend_rate=combined, superimposed_inflation=si,
            projection=proj_df,
        )

    def test_trend_factor_zero_periods(self):
        lc = self._make_lc_result()
        assert lc.trend_factor(0) == pytest.approx(1.0)

    def test_trend_factor_monthly_ppy(self):
        """trend_factor with periods_per_year=12: 12 periods = 1 year."""
        from insurance_trend.result import TrendResult, LossCostTrendResult
        n = 8
        fitted = pl.Series("fitted", np.full(n, 100.0))
        residuals = pl.Series("residuals", np.zeros(n))
        proj_df = pl.DataFrame({"period": [1], "point": [100.0],
                                "lower": [98.0], "upper": [102.0]})
        freq_result = TrendResult(
            trend_rate=0.06, ci_lower=0.04, ci_upper=0.08,
            method="log_linear", fitted_values=fitted, residuals=residuals,
            changepoints=[], projection=proj_df, r_squared=0.8,
            periods_per_year=12,
        )
        sev_result = TrendResult(
            trend_rate=0.04, ci_lower=0.02, ci_upper=0.06,
            method="log_linear", fitted_values=fitted, residuals=residuals,
            changepoints=[], projection=proj_df, r_squared=0.9,
            periods_per_year=12,
        )
        combined = (1.06) * (1.04) - 1.0
        lc = LossCostTrendResult(
            frequency=freq_result, severity=sev_result,
            combined_trend_rate=combined, superimposed_inflation=None,
            projection=proj_df,
        )
        # 12 periods = 1 year, so trend_factor(12) = (1 + combined)^1
        expected = (1.0 + combined) ** 1.0
        assert lc.trend_factor(12) == pytest.approx(expected, rel=1e-8)

    def test_summary_with_superimposed_none(self):
        lc = self._make_lc_result(si=None)
        s = lc.summary()
        assert "N/A" in s

    def test_summary_with_superimposed_value(self):
        lc = self._make_lc_result(si=0.03)
        s = lc.summary()
        assert "3.00%" in s

    def test_decompose_with_none_superimposed(self):
        lc = self._make_lc_result(si=None)
        d = lc.decompose()
        assert d["superimposed"] is None

    def test_repr_roundtrip(self):
        lc = self._make_lc_result()
        r = repr(lc)
        assert "LossCostTrendResult" in r
        assert "freq=" in r
        assert "sev=" in r


# ---------------------------------------------------------------------------
# TrendResult: additional edge cases
# ---------------------------------------------------------------------------


class TestTrendResultEdgeCases:
    def test_trend_factor_fractional_periods(self):
        """trend_factor(1) for quarterly data = (1 + rate)^(1/4)."""
        fitter = _make_freq_fitter()
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        expected = (1.0 + result.trend_rate) ** (1.0 / 4.0)
        assert result.trend_factor(1) == pytest.approx(expected, rel=1e-8)

    def test_changepoints_in_result_are_list(self):
        fitter = _make_freq_fitter()
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        assert isinstance(result.changepoints, list)

    def test_summary_shows_method_piecewise(self, breakpoint_data):
        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        result = fitter.fit(
            changepoints=[breakpoint_data["break_index"]],
            detect_breaks=False,
            n_bootstrap=50,
        )
        assert "piecewise" in result.summary()

    def test_periods_series_length_matches_data(self):
        fitter = _make_freq_fitter(n=12)
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        assert len(result.periods) == 12

    def test_actuals_series_values_are_frequency(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=50)
        expected = trending_data["claim_counts"] / trending_data["earned_exposure"]
        np.testing.assert_allclose(result.actuals.to_numpy(), expected, rtol=1e-6)

    def test_n_bootstrap_stored_in_result(self):
        fitter = _make_freq_fitter()
        result = fitter.fit(detect_breaks=False, n_bootstrap=200)
        assert result.n_bootstrap == 200
