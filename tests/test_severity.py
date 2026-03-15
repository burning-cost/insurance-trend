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


# ------------------------------------------------------------------ #
# Regression tests for P0/P1 bugs
# ------------------------------------------------------------------ #

class TestRegressionBugs:
    """Regression tests introduced to guard the P0/P1 bug fixes."""

    def test_p0_1_superimposed_inflation_not_double_deflated(self):
        """P0-1: superimposed_inflation() must equal the trend of the deflated series.

        When the external index grows at ~4% pa and the true underlying severity
        trend (after deflation) is ~3% pa, superimposed_inflation() should return
        approximately 3%, NOT (1.03)/(1.04)-1 ≈ -0.96%.
        """
        rng = np.random.default_rng(99)
        n = 20
        t = np.arange(n, dtype=float)
        # External index: 4% pa
        index_pa = 0.04
        index = np.exp(index_pa / 4 * t) * np.exp(rng.normal(0, 0.005, n))
        # True SI: 3% pa above index
        si_pa = 0.03
        # Raw severity = index * superimposed component
        raw_sev = 5000.0 * index * np.exp(si_pa / 4 * t) * np.exp(rng.normal(0, 0.01, n))
        claim_counts = np.full(n, 100.0)
        total_paid = raw_sev * claim_counts

        periods = [f"{y}Q{q}" for y in range(2019, 2024) for q in range(1, 5)][:n]
        fitter = SeverityTrendFitter(
            periods=periods,
            total_paid=total_paid,
            claim_counts=claim_counts,
            external_index=pl.Series("idx", index),
        )
        fitter.fit(detect_breaks=False, seasonal=False, n_bootstrap=100)
        si = fitter.superimposed_inflation()
        assert si is not None
        # Must be close to the true 3% SI — NOT the double-deflated ~-0.96%
        assert si > 0.01, f"SI should be ~3% pa, got {si:.4f}. Double-deflation bug?"
        assert abs(si - si_pa) < 0.04, f"SI {si:.4f} too far from true {si_pa:.4f}"

    def test_p0_1_superimposed_inflation_equals_deflated_trend(self):
        """P0-1: superimposed_inflation() == fitted_result.trend_rate (by definition)."""
        rng = np.random.default_rng(7)
        n = 16
        t = np.arange(n, dtype=float)
        index = np.exp(0.04 / 4 * t) * np.exp(rng.normal(0, 0.003, n))
        raw_sev = 4000.0 * index * np.exp(0.05 / 4 * t) * np.exp(rng.normal(0, 0.01, n))
        counts = np.full(n, 80.0)
        periods = [f"{y}Q{q}" for y in range(2020, 2024) for q in range(1, 5)]
        fitter = SeverityTrendFitter(
            periods=periods,
            total_paid=raw_sev * counts,
            claim_counts=counts,
            external_index=pl.Series("idx", index),
        )
        result = fitter.fit(detect_breaks=False, seasonal=False, n_bootstrap=50)
        si = fitter.superimposed_inflation()
        # SI is the trend of the deflated series — must equal result.trend_rate exactly
        assert si is not None
        assert abs(si - result.trend_rate) < 1e-10, (
            f"SI {si} != trend_rate {result.trend_rate}. Returned wrong value."
        )

    def test_p0_2_piecewise_bootstrap_ci_not_inflated(self, breakpoint_data):
        """P0-2: bootstrap CI width must not be ~37x wider when breaks exist.

        With a clean structural break and piecewise fit, the CI on the final
        segment trend should be narrow. Previously, full-series OLS residuals
        were used, making CIs absurdly wide.
        """
        fitter = SeverityTrendFitter(
            periods=breakpoint_data["periods"],
            total_paid=breakpoint_data["total_paid"],
            claim_counts=breakpoint_data["claim_counts"],
        )
        result = fitter.fit(
            changepoints=[breakpoint_data["break_index"]],
            detect_breaks=False,
            seasonal=False,
            n_bootstrap=200,
        )
        ci_width = result.ci_upper - result.ci_lower
        # A CI of ±80 pp or wider is a symptom of the wrong-residuals bug.
        # A reasonable CI for a clean synthetic break should be well under 40 pp.
        assert ci_width < 0.40, (
            f"CI width {ci_width:.4f} is too wide. "
            "Bootstrap may still be using full-series OLS residuals."
        )

    def test_p0_2_freq_piecewise_bootstrap_ci_not_inflated(self, breakpoint_data):
        """P0-2 (frequency): same check for FrequencyTrendFitter."""
        from insurance_trend import FrequencyTrendFitter

        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        result = fitter.fit(
            changepoints=[breakpoint_data["break_index"]],
            detect_breaks=False,
            seasonal=False,
            n_bootstrap=200,
        )
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width < 0.40, (
            f"Frequency CI width {ci_width:.4f} too wide. "
            "Bootstrap may still be using full-series OLS residuals."
        )

    def test_p0_3_seasonal_phase_not_reset_at_break(self, breakpoint_data):
        """P0-3: seasonal dummies must not reset to Q1 at each breakpoint.

        Fit with and without an explicit mid-series break. Without the bug,
        the fitted values in Q1 positions should be consistent regardless of
        which segment they fall in. As a proxy: fitting with the correct global
        t should produce lower in-sample residuals than local t.
        """
        import warnings
        from insurance_trend import FrequencyTrendFitter

        fitter = FrequencyTrendFitter(
            periods=breakpoint_data["periods"],
            claim_counts=breakpoint_data["claim_counts"],
            earned_exposure=breakpoint_data["earned_exposure"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fitter.fit(
                changepoints=[breakpoint_data["break_index"]],
                detect_breaks=False,
                seasonal=True,
                n_bootstrap=50,
            )
        # Residuals should be small (not systematically biased by wrong phase)
        residuals = result.residuals.to_numpy()
        max_abs_resid = float(np.max(np.abs(residuals)))
        assert max_abs_resid < 0.30, (
            f"Max residual {max_abs_resid:.4f} suspiciously large. "
            "Seasonal phase may be wrong at segment boundaries."
        )

    def test_p1_1_plot_docstring_no_mention_90pct(self):
        """P1-1: TrendResult.plot docstring must not promise both 90% and 95% bands."""
        import inspect
        from insurance_trend.result import TrendResult
        doc = inspect.getdoc(TrendResult.plot)
        # The old bug: docstring said "90 % and 95 % CI bands" while only one was drawn.
        # After the fix the docstring should not claim two distinct CI bands.
        assert "90 %" not in doc and "90%" not in doc, (
            "Docstring still mentions 90% CI band but only one band is drawn."
        )

    def test_p1_2_local_linear_bootstrap_cap_warns(self):
        """P1-2: requesting >200 bootstrap reps for local_linear_trend must warn."""
        from insurance_trend import FrequencyTrendFitter
        import warnings

        rng = np.random.default_rng(5)
        n = 12
        counts = np.round(1000.0 * 0.1 * np.ones(n)).astype(float)
        exposure = np.full(n, 1000.0)
        periods = [f"2021Q{q}" for q in range(1, 5)] + \
                  [f"2022Q{q}" for q in range(1, 5)] + \
                  [f"2023Q{q}" for q in range(1, 5)]

        fitter = FrequencyTrendFitter(
            periods=periods,
            claim_counts=counts,
            earned_exposure=exposure,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter.fit(method="local_linear_trend", n_bootstrap=500)

        warning_messages = [str(w.message) for w in caught]
        assert any("200" in msg or "capped" in msg.lower() for msg in warning_messages), (
            "Expected a warning about bootstrap cap at 200 for local_linear_trend. "
            f"Got: {warning_messages}"
        )
