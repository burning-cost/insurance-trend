"""Tests for LossCostTrendFitter."""

import numpy as np
import polars as pl
import pytest

from insurance_trend import LossCostTrendFitter, LossCostTrendResult


class TestLossCostTrendFitterConstruction:
    def test_basic_construction(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        assert fitter is not None

    def test_loss_cost_computed_correctly(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        expected = trending_data["total_paid"] / trending_data["earned_exposure"]
        np.testing.assert_allclose(fitter.loss_cost, expected)

    def test_mismatched_lengths_raises(self, trending_data):
        with pytest.raises(ValueError, match="same length"):
            LossCostTrendFitter(
                periods=trending_data["periods"],
                claim_counts=trending_data["claim_counts"][:-1],
                earned_exposure=trending_data["earned_exposure"],
                total_paid=trending_data["total_paid"],
            )

    def test_summary_string(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        s = fitter.summary()
        assert "LossCostTrendFitter" in s


class TestLossCostFit:
    def test_returns_loss_cost_trend_result(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, LossCostTrendResult)

    def test_has_frequency_and_severity_results(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        from insurance_trend import TrendResult
        assert isinstance(result.frequency, TrendResult)
        assert isinstance(result.severity, TrendResult)

    def test_combined_trend_rate_equals_freq_times_sev(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        expected = (1.0 + result.frequency.trend_rate) * (1.0 + result.severity.trend_rate) - 1.0
        assert abs(result.combined_trend_rate - expected) < 1e-10

    def test_combined_trend_positive(self, trending_data):
        """With -2% freq and +8% sev, combined should be ~+5.8%."""
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False, seasonal=False)
        assert result.combined_trend_rate > 0, "Expected positive combined trend"

    def test_superimposed_is_none_without_index(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        assert result.superimposed_inflation is None

    def test_superimposed_is_float_with_index(self, trending_data, external_index_series):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
            external_index=external_index_series,
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.superimposed_inflation, float)

    def test_projection_is_polars_dataframe(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert isinstance(result.projection, pl.DataFrame)

    def test_projection_has_correct_columns(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert set(result.projection.columns) == {"period", "point", "lower", "upper"}


class TestLossCostDecompose:
    def test_decompose_returns_dict(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        d = result.decompose()
        assert isinstance(d, dict)

    def test_decompose_keys(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        d = result.decompose()
        assert set(d.keys()) == {"freq_trend", "sev_trend", "combined_trend", "superimposed"}

    def test_decompose_combined_matches_attribute(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        d = result.decompose()
        assert abs(d["combined_trend"] - result.combined_trend_rate) < 1e-10

    def test_decompose_superimposed_none_without_index(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        d = result.decompose()
        assert d["superimposed"] is None

    def test_trend_factor_two_years(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        expected = (1.0 + result.combined_trend_rate) ** 2.0
        assert abs(result.trend_factor(8) - expected) < 1e-8


class TestProjectedLossCost:
    def test_projected_loss_cost_returns_dataframe(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        proj = fitter.projected_loss_cost(future_periods=8)
        assert isinstance(proj, pl.DataFrame)

    def test_projected_loss_cost_length(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        proj = fitter.projected_loss_cost(future_periods=6)
        assert len(proj) == 6

    def test_projected_lower_less_than_point(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        proj = fitter.projected_loss_cost(future_periods=8)
        lower = proj["lower"].to_numpy()
        point = proj["point"].to_numpy()
        upper = proj["upper"].to_numpy()
        assert np.all(lower <= point + 1e-6)
        assert np.all(point <= upper + 1e-6)


class TestLossCostSummaryAndRepr:
    def test_summary_string(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        s = result.summary()
        assert "Loss Cost Trend" in s

    def test_repr(self, trending_data):
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        r = repr(result)
        assert "LossCostTrendResult" in r

    def test_plot_returns_figure(self, trending_data):
        import matplotlib
        matplotlib.use("Agg")
        fitter = LossCostTrendFitter(
            periods=trending_data["periods"],
            claim_counts=trending_data["claim_counts"],
            earned_exposure=trending_data["earned_exposure"],
            total_paid=trending_data["total_paid"],
        )
        result = fitter.fit(detect_breaks=False)
        fig = result.plot()
        assert fig is not None
