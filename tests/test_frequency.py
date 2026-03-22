"""Tests for FrequencyTrendFitter."""

import numpy as np
import polars as pl
import pandas as pd
import pytest

from insurance_trend import FrequencyTrendFitter, TrendResult


class TestFrequencyTrendFitterConstruction:
    def test_basic_construction(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        assert fitter is not None

    def test_frequency_computed_correctly(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        expected = trending_data["claim_counts"] / trending_data["earned_exposure"]
        np.testing.assert_allclose(fitter.frequency, expected)

    def test_polars_series_input(self, polars_trending_data):
        fitter = FrequencyTrendFitter(
            periods=polars_trending_data["periods"],
            claim_counts=polars_trending_data["claim_counts"],
            earned_exposure=polars_trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_pandas_series_input(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=pd.Series(trending_data["periods"]),
            claim_counts=pd.Series(trending_data["claim_counts"]),
            earned_exposure=pd.Series(trending_data["earned_exposure"]),
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_list_input(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=list(trending_data["periods"]),
            claim_counts=list(trending_data["claim_counts"]),
            earned_exposure=list(trending_data["earned_exposure"]),
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_mismatched_lengths_raises(self, trending_data):
        with pytest.raises(ValueError, match="same length"):
            FrequencyTrendFitter(
                periods=trending_data["periods"],
                claim_counts=trending_data["claim_counts"][:-1],
                earned_exposure=trending_data["earned_exposure"],
            )

    def test_zero_exposure_raises(self, trending_data):
        exposure = trending_data["earned_exposure"].copy()
        exposure[3] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            FrequencyTrendFitter(
                periods=trending_data["periods"],
                claim_counts=trending_data["claim_counts"],
                earned_exposure=exposure,
            )

    def test_weights_accepted(self, trending_data):
        n = len(trending_data["periods"])
        weights = np.linspace(0.5, 1.0, n)
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            weights=weights,
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_summary_string(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        s = fitter.summary()
        assert "FrequencyTrendFitter" in s


class TestFrequencyFitLogLinear:
    def test_returns_trend_result(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_trend_rate_is_float(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.trend_rate, float)

    def test_negative_trend_recovered(self, trending_data):
        """Should detect approximately -2% pa frequency trend."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, seasonal=False)
        # Allow wide tolerance due to synthetic noise
        assert result.trend_rate < 0, "Expected negative frequency trend"
        assert result.trend_rate > -0.20, "Trend too strongly negative"

    def test_method_label_log_linear(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(method="log_linear", detect_breaks=False)
        assert result.method == "log_linear"

    def test_ci_ordering(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=100)
        assert result.ci_lower <= result.trend_rate <= result.ci_upper

    def test_r_squared_range(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert 0.0 <= result.r_squared <= 1.0

    def test_fitted_values_polars_series(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.fitted_values, pl.Series)

    def test_residuals_polars_series(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.residuals, pl.Series)

    def test_residuals_length(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert len(result.residuals) == len(trending_data["periods"])

    def test_projection_is_dataframe(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert isinstance(result.projection, pl.DataFrame)

    def test_projection_columns(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert set(result.projection.columns) == {"period", "point", "lower", "upper"}

    def test_projection_length(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert len(result.projection) == 8

    def test_projection_monotone_for_negative_trend(self, trending_data):
        """With negative trend, projected point values should decrease."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, seasonal=False)
        if result.trend_rate < 0:
            pts = result.projection["point"].to_numpy()
            assert pts[-1] < pts[0]

    def test_no_seasonal_dummies(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, seasonal=False)
        assert isinstance(result, TrendResult)

    def test_flat_data_near_zero_trend(self, flat_trend_data):
        fitter = FrequencyTrendFitter(
            periods=flat_trend_data["periods"],
            claim_counts=flat_trend_data["claim_counts"],
            earned_exposure=flat_trend_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert abs(result.trend_rate) < 0.10, f"Expected near-zero trend, got {result.trend_rate}"

    def test_invalid_method_raises(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        with pytest.raises(ValueError, match="Unknown method"):
            fitter.fit(method="random_forest")

    def test_short_data(self, short_data):
        """Should handle 6 periods without crashing."""
        fitter = FrequencyTrendFitter(
            periods=short_data["periods"],
            claim_counts=short_data["claim_counts"],
            earned_exposure=short_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)


class TestFrequencyFitPiecewise:
    def test_piecewise_fit_with_explicit_changepoint(self, breakpoint_data):
        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        result = fitter.fit(changepoints=[breakpoint_data["break_index"]], detect_breaks=False)
        assert result.method == "piecewise"

    def test_piecewise_result_has_changepoints(self, breakpoint_data):
        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        result = fitter.fit(changepoints=[breakpoint_data["break_index"]])
        assert breakpoint_data["break_index"] in result.changepoints

    def test_detect_breaks_warns(self, breakpoint_data):
        """Auto-detection of a clear structural break should emit a UserWarning."""
        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        with pytest.warns(UserWarning):
            fitter.fit(detect_breaks=True, penalty=0.5)

    def test_detect_breaks_false_no_warning(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitter.fit(detect_breaks=False)


class TestTrendResultMethods:
    def test_trend_factor_identity(self, trending_data):
        """trend_factor(0) should equal 1.0 (zero periods = no change)."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        assert abs(result.trend_factor(0) - 1.0) < 1e-10

    def test_trend_factor_one_year(self, trending_data):
        """trend_factor(4 quarters) should equal (1 + annual_rate)^1."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        expected = (1.0 + result.trend_rate) ** 1.0
        assert abs(result.trend_factor(4) - expected) < 1e-8

    def test_trend_factor_two_years(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        expected = (1.0 + result.trend_rate) ** 2.0
        assert abs(result.trend_factor(8) - expected) < 1e-8

    def test_summary_contains_trend_rate(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        s = result.summary()
        assert "Trend rate" in s

    def test_repr_contains_trend_rate(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        r = repr(result)
        assert "trend_rate" in r

    def test_plot_returns_figure(self, trending_data):
        import matplotlib
        matplotlib.use("Agg")
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False)
        fig = result.plot()
        assert fig is not None

    def test_periods_per_year_zero_raises(self, trending_data):
        """periods_per_year=0 causes ZeroDivisionError downstream — catch it early."""
        with pytest.raises(ValueError, match="periods_per_year"):
            FrequencyTrendFitter(
                periods=trending_data["periods"],
                claim_counts=trending_data["claim_counts"],
                earned_exposure=trending_data["earned_exposure"],
                periods_per_year=0,
            )

    def test_periods_per_year_negative_raises(self, trending_data):
        with pytest.raises(ValueError, match="periods_per_year"):
            FrequencyTrendFitter(
                periods=trending_data["periods"],
                claim_counts=trending_data["claim_counts"],
                earned_exposure=trending_data["earned_exposure"],
                periods_per_year=-4,
            )

    def test_periods_per_year_positive_does_not_raise(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            periods_per_year=12,
        )
        assert fitter._periods_per_year == 12

    def test_ci_level_zero_raises(self, trending_data):
        """ci_level=0 is not a valid confidence level."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        with pytest.raises(ValueError, match="ci_level"):
            fitter.fit(detect_breaks=False, ci_level=0.0)

    def test_ci_level_one_raises(self, trending_data):
        """ci_level=1 is not a valid confidence level."""
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        with pytest.raises(ValueError, match="ci_level"):
            fitter.fit(detect_breaks=False, ci_level=1.0)

    def test_ci_level_negative_raises(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        with pytest.raises(ValueError, match="ci_level"):
            fitter.fit(detect_breaks=False, ci_level=-0.5)

    def test_ci_level_valid_does_not_raise(self, trending_data):
        fitter = FrequencyTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
        )
        result = fitter.fit(detect_breaks=False, ci_level=0.90, n_bootstrap=50)
        assert result is not None
