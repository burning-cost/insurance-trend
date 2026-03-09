"""Tests for SeverityTrendFitter."""

import numpy as np
import polars as pl
import pandas as pd
import pytest

from insurance_trend import SeverityTrendFitter, TrendResult


class TestSeverityTrendFitterConstruction:
    def test_basic_construction(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        assert fitter is not None

    def test_severity_computed_correctly(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        expected = trending_data["total_paid"] / trending_data["claim_counts"]
        np.testing.assert_allclose(fitter.severity, expected)

    def test_polars_input(self, polars_trending_data):
        fitter = SeverityTrendFitter(
            periods=polars_trending_data["periods"],
            total_paid=polars_trending_data["total_paid"],
            claim_counts=polars_trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_pandas_input(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=pd.Series(trending_data["periods"]),
            total_paid=pd.Series(trending_data["total_paid"]),
            claim_counts=pd.Series(trending_data["claim_counts"]),
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_mismatched_lengths_raises(self, trending_data):
        with pytest.raises(ValueError, match="same length"):
            SeverityTrendFitter(
                periods=trending_data["periods"],
                total_paid=trending_data["total_paid"][:-2],
                claim_counts=trending_data["claim_counts"],
            )

    def test_zero_claim_counts_raises(self, trending_data):
        counts = trending_data["claim_counts"].copy()
        counts[2] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            SeverityTrendFitter(
                periods=trending_data["periods"],
                total_paid=trending_data["total_paid"],
                claim_counts=counts,
            )

    def test_zero_total_paid_raises(self, trending_data):
        paid = trending_data["total_paid"].copy()
        paid[1] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            SeverityTrendFitter(
                periods=trending_data["periods"],
                total_paid=paid,
                claim_counts=trending_data["claim_counts"],
            )

    def test_external_index_accepted(self, trending_data, external_index_series):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        assert fitter.deflated_severity is not None

    def test_no_external_index_deflated_is_none(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        assert fitter.deflated_severity is None

    def test_summary_string(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        s = fitter.summary()
        assert "SeverityTrendFitter" in s


class TestSeverityFitLogLinear:
    def test_returns_trend_result(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result, TrendResult)

    def test_positive_trend_recovered(self, trending_data):
        """Should detect approximately +8% pa severity trend."""
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False, seasonal=False)
        assert result.trend_rate > 0, f"Expected positive severity trend, got {result.trend_rate}"
        assert result.trend_rate < 0.25, f"Trend too high: {result.trend_rate}"

    def test_ci_ordering(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False, n_bootstrap=100)
        assert result.ci_lower <= result.trend_rate <= result.ci_upper

    def test_r_squared_range(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False)
        assert 0.0 <= result.r_squared <= 1.0

    def test_fitted_values_polars(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.fitted_values, pl.Series)

    def test_residuals_polars(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False)
        assert isinstance(result.residuals, pl.Series)

    def test_projection_columns(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=8)
        assert set(result.projection.columns) == {"period", "point", "lower", "upper"}

    def test_projection_length(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(detect_breaks=False, projection_periods=6)
        assert len(result.projection) == 6

    def test_method_label(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        result = fitter.fit(method="log_linear", detect_breaks=False)
        assert result.method == "log_linear"

    def test_piecewise_method_label(self, breakpoint_data):
        fitter = SeverityTrendFitter(
            periods=breakpoint_data["periods"],
            total_paid=breakpoint_data["total_paid"],
            claim_counts=breakpoint_data["claim_counts"],
        )
        result = fitter.fit(changepoints=[breakpoint_data["break_index"]])
        assert result.method == "piecewise"

    def test_invalid_method_raises(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        with pytest.raises(ValueError, match="Unknown method"):
            fitter.fit(method="gbm")


class TestSuperimposedInflation:
    def test_no_index_superimposed_is_none_before_fit(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        assert fitter.superimposed_inflation() is None

    def test_no_index_superimposed_is_none_after_fit(self, trending_data):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
        )
        fitter.fit(detect_breaks=False)
        assert fitter.superimposed_inflation() is None

    def test_with_index_superimposed_is_float(self, trending_data, external_index_series):
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        fitter.fit(detect_breaks=False)
        si = fitter.superimposed_inflation()
        assert isinstance(si, float)

    def test_superimposed_less_than_total_trend(self, trending_data, external_index_series):
        """When external index has positive trend, SI < total severity trend."""
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        result = fitter.fit(detect_breaks=False)
        si = fitter.superimposed_inflation()
        # SI = total_trend - index_trend; index_trend ~4%, total_trend ~8%
        # So SI should be less than total_trend (and may be positive or near zero)
        assert si < result.trend_rate + 0.01  # allow small numeric tolerance

    def test_deflated_severity_is_lower(self, trending_data, external_index_series):
        """Deflated severity should be on a lower scale than raw severity."""
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        raw = fitter.severity
        deflated = fitter.deflated_severity
        assert deflated is not None
        # Since index grows over time, deflated severity in last period < raw
        assert deflated[-1] < raw[-1]

    def test_external_index_rebased_to_one(self, trending_data, external_index_series):
        """The external index should be internally rebased so index[0] = 1.0."""
        fitter = SeverityTrendFitter(
            periods=trending_data["periods"],
            total_paid=trending_data["total_paid"],
            claim_counts=trending_data["claim_counts"],
            external_index=external_index_series,
        )
        # Internal rebase means deflated_severity[0] == severity[0] / 1.0 = severity[0]
        np.testing.assert_allclose(fitter.deflated_severity[0], fitter.severity[0], rtol=1e-6)

    def test_external_index_shorter_than_periods_raises(self, trending_data, external_index_series):
        """Index shorter than the number of periods should raise ValueError."""
        short_index = external_index_series.head(5)
        with pytest.raises(ValueError, match="shorter than"):
            SeverityTrendFitter(
                periods=trending_data["periods"],
                total_paid=trending_data["total_paid"],
                claim_counts=trending_data["claim_counts"],
                external_index=short_index,
            )
