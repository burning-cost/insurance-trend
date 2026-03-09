"""Tests for TrendResult and LossCostTrendResult dataclasses."""

import numpy as np
import polars as pl
import pytest

from insurance_trend.result import LossCostTrendResult, TrendResult


def _make_projection(n: int = 8) -> pl.DataFrame:
    """Helper to build a valid projection DataFrame."""
    t = list(range(1, n + 1))
    base = 100.0
    rate = 0.02
    return pl.DataFrame({
        "period": t,
        "point": [base * (1 + rate) ** i for i in range(1, n + 1)],
        "lower": [base * (1 + rate * 0.5) ** i for i in range(1, n + 1)],
        "upper": [base * (1 + rate * 1.5) ** i for i in range(1, n + 1)],
    })


def _make_result(trend_rate: float = 0.05) -> TrendResult:
    """Helper to construct a TrendResult for testing."""
    n = 12
    actuals = np.full(n, 100.0) * np.exp(
        np.cumsum(np.random.default_rng(0).normal(0, 0.02, n))
    )
    fitted = actuals * 1.01
    residuals = actuals / fitted - 1.0
    return TrendResult(
        trend_rate=trend_rate,
        ci_lower=trend_rate - 0.02,
        ci_upper=trend_rate + 0.02,
        method="log_linear",
        fitted_values=pl.Series("fitted", fitted),
        residuals=pl.Series("residuals", residuals),
        changepoints=[],
        projection=_make_projection(),
        r_squared=0.85,
        actuals=pl.Series("actuals", actuals),
        periods=pl.Series("periods", [f"P{i}" for i in range(n)]),
        n_bootstrap=1000,
        periods_per_year=4,
    )


class TestTrendResultAttributes:
    def test_trend_rate_stored(self):
        r = _make_result(0.08)
        assert r.trend_rate == 0.08

    def test_ci_stored(self):
        r = _make_result()
        assert r.ci_lower < r.trend_rate < r.ci_upper

    def test_method_stored(self):
        r = _make_result()
        assert r.method == "log_linear"

    def test_fitted_values_is_polars(self):
        r = _make_result()
        assert isinstance(r.fitted_values, pl.Series)

    def test_residuals_is_polars(self):
        r = _make_result()
        assert isinstance(r.residuals, pl.Series)

    def test_projection_is_polars_df(self):
        r = _make_result()
        assert isinstance(r.projection, pl.DataFrame)

    def test_r_squared_stored(self):
        r = _make_result()
        assert r.r_squared == 0.85

    def test_changepoints_stored(self):
        r = _make_result()
        assert r.changepoints == []


class TestTrendFactor:
    def test_zero_periods(self):
        r = _make_result(0.10)
        assert abs(r.trend_factor(0) - 1.0) < 1e-10

    def test_one_year(self):
        r = _make_result(0.10)
        assert abs(r.trend_factor(4) - 1.10) < 1e-8

    def test_two_years(self):
        r = _make_result(0.10)
        assert abs(r.trend_factor(8) - 1.10 ** 2) < 1e-8

    def test_half_year(self):
        r = _make_result(0.10)
        expected = (1.10) ** 0.5
        assert abs(r.trend_factor(2) - expected) < 1e-8

    def test_18_months(self):
        r = _make_result(0.10)
        expected = (1.10) ** 1.5
        assert abs(r.trend_factor(6) - expected) < 1e-8

    def test_negative_trend_factor_less_than_one(self):
        r = _make_result(-0.05)
        assert r.trend_factor(4) < 1.0

    def test_zero_trend_factor_always_one(self):
        r = _make_result(0.0)
        assert abs(r.trend_factor(4) - 1.0) < 1e-10
        assert abs(r.trend_factor(8) - 1.0) < 1e-10


class TestTrendResultSummaryAndRepr:
    def test_summary_is_string(self):
        r = _make_result()
        assert isinstance(r.summary(), str)

    def test_summary_contains_trend_rate(self):
        r = _make_result(0.085)
        s = r.summary()
        assert "8.50%" in s

    def test_summary_contains_method(self):
        r = _make_result()
        assert "log_linear" in r.summary()

    def test_repr_contains_class_name(self):
        r = _make_result()
        assert "TrendResult" in repr(r)

    def test_repr_contains_trend_rate(self):
        r = _make_result(0.05)
        assert "0.0500" in repr(r)


class TestLossCostTrendResult:
    def _make_lc_result(self) -> LossCostTrendResult:
        freq = _make_result(-0.02)
        sev = _make_result(0.08)
        combined = (1 + freq.trend_rate) * (1 + sev.trend_rate) - 1.0
        return LossCostTrendResult(
            frequency=freq,
            severity=sev,
            combined_trend_rate=combined,
            superimposed_inflation=0.04,
            projection=_make_projection(),
        )

    def test_decompose_keys(self):
        lc = self._make_lc_result()
        d = lc.decompose()
        assert set(d.keys()) == {"freq_trend", "sev_trend", "combined_trend", "superimposed"}

    def test_decompose_values_correct(self):
        lc = self._make_lc_result()
        d = lc.decompose()
        assert abs(d["freq_trend"] - (-0.02)) < 1e-10
        assert abs(d["sev_trend"] - 0.08) < 1e-10

    def test_combined_trend_correct(self):
        lc = self._make_lc_result()
        expected = (1 - 0.02) * (1 + 0.08) - 1.0
        assert abs(lc.combined_trend_rate - expected) < 1e-10

    def test_trend_factor_two_years(self):
        lc = self._make_lc_result()
        expected = (1 + lc.combined_trend_rate) ** 2.0
        assert abs(lc.trend_factor(8) - expected) < 1e-8

    def test_summary_is_string(self):
        lc = self._make_lc_result()
        assert isinstance(lc.summary(), str)

    def test_summary_contains_combined_trend(self):
        lc = self._make_lc_result()
        s = lc.summary()
        assert "Combined" in s

    def test_repr_contains_class_name(self):
        lc = self._make_lc_result()
        assert "LossCostTrendResult" in repr(lc)

    def test_superimposed_in_decompose(self):
        lc = self._make_lc_result()
        d = lc.decompose()
        assert d["superimposed"] == 0.04
