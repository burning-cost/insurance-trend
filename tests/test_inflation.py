"""Tests for InflationDecomposer and InflationDecompositionResult.

The test suite covers:
- Construction validation (good path + all error paths)
- Fit correctness on synthetic data with known structural/cyclical properties
- Result dataclass attributes and methods
- Edge cases (no cycle, with seasonal, monthly data, pandas input)
- Numerical properties (component reconstruction, rate sign)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_trend.inflation import InflationDecomposer, InflationDecompositionResult


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_periods(n: int, start_year: int = 2015) -> list[str]:
    """Generate n quarterly period labels."""
    labels = []
    for year in range(start_year, start_year + 20):
        for q in range(1, 5):
            labels.append(f"{year}Q{q}")
            if len(labels) == n:
                return labels
    return labels


def _make_series(
    n: int = 40,
    structural_pa: float = 0.07,
    cycle_amplitude: float = 0.05,
    cycle_period_years: float = 6.0,
    noise_sigma: float = 0.008,
    seed: int = 42,
    periods_per_year: int = 4,
    base: float = 100.0,
) -> np.ndarray:
    """Synthetic index with known structural trend and stochastic cycle.

    Constructed as:
        log(y_t) = structural_pa/ppy * t + cycle_amplitude * sin(2*pi*t / cycle_period_periods) + noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    structural = structural_pa / periods_per_year * t
    cycle_period_periods = cycle_period_years * periods_per_year
    cycle = cycle_amplitude * np.sin(2.0 * np.pi * t / cycle_period_periods)
    noise = rng.normal(0, noise_sigma, n)
    return base * np.exp(structural + cycle + noise)


# Shared fixture for a clean 40-period series
@pytest.fixture(scope="module")
def clean_series():
    return _make_series(n=40, structural_pa=0.07, cycle_amplitude=0.04, seed=1)


@pytest.fixture(scope="module")
def clean_periods():
    return _make_periods(40)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestInflationDecomposerConstruction:
    def test_basic_construction_list_input(self):
        series = list(_make_series(n=24))
        decomposer = InflationDecomposer(series=series, periods=_make_periods(24))
        assert decomposer is not None

    def test_basic_construction_numpy_input(self):
        series = _make_series(n=24)
        decomposer = InflationDecomposer(series=series)
        assert decomposer is not None

    def test_polars_series_input(self):
        series = _make_series(n=24)
        decomposer = InflationDecomposer(
            series=pl.Series("sev", series),
            periods=pl.Series("p", _make_periods(24)),
        )
        assert decomposer is not None

    def test_pandas_series_with_index_as_periods(self):
        """Pandas Series with string index should use the index as period labels."""
        series = _make_series(n=24)
        periods = _make_periods(24)
        s = pd.Series(series, index=periods)
        decomposer = InflationDecomposer(series=s)
        assert decomposer is not None

    def test_pandas_series_with_explicit_periods(self):
        """Explicit periods override pandas index."""
        series = _make_series(n=24)
        s = pd.Series(series)
        decomposer = InflationDecomposer(series=s, periods=_make_periods(24))
        assert decomposer is not None

    def test_non_positive_values_log_transform_raises(self):
        series = _make_series(n=24)
        bad = series.copy()
        bad[5] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            InflationDecomposer(series=bad, log_transform=True)

    def test_negative_value_log_transform_raises(self):
        series = _make_series(n=24)
        bad = series.copy()
        bad[3] = -50.0
        with pytest.raises(ValueError, match="non-positive"):
            InflationDecomposer(series=bad, log_transform=True)

    def test_zero_periods_per_year_raises(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            InflationDecomposer(series=_make_series(n=24), periods_per_year=0)

    def test_negative_periods_per_year_raises(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            InflationDecomposer(series=_make_series(n=24), periods_per_year=-1)

    def test_too_short_series_with_cycle_raises(self):
        short_series = _make_series(n=10)
        with pytest.raises(ValueError, match="cycle=True"):
            InflationDecomposer(series=short_series, cycle=True)

    def test_too_short_series_without_cycle_raises(self):
        short_series = _make_series(n=4)
        with pytest.raises(ValueError, match="at least"):
            InflationDecomposer(series=short_series, cycle=False)

    def test_invalid_cycle_period_bounds_raises(self):
        with pytest.raises(ValueError, match="cycle_period_bounds"):
            InflationDecomposer(
                series=_make_series(n=24),
                cycle_period_bounds=(8.0, 2.0),  # lower > upper
            )

    def test_equal_cycle_period_bounds_raises(self):
        with pytest.raises(ValueError, match="cycle_period_bounds"):
            InflationDecomposer(
                series=_make_series(n=24),
                cycle_period_bounds=(4.0, 4.0),
            )

    def test_non_positive_cycle_period_bound_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            InflationDecomposer(
                series=_make_series(n=24),
                cycle_period_bounds=(0.0, 6.0),
            )

    def test_seasonal_less_than_2_raises(self):
        with pytest.raises(ValueError, match="seasonal"):
            InflationDecomposer(series=_make_series(n=24), seasonal=1)

    def test_periods_length_mismatch_raises(self):
        series = _make_series(n=24)
        with pytest.raises(ValueError, match="periods length"):
            InflationDecomposer(series=series, periods=_make_periods(20))

    def test_16_observations_minimum_with_cycle_accepted(self):
        """Exactly 16 observations with cycle=True should be accepted."""
        series = _make_series(n=16)
        decomposer = InflationDecomposer(series=series, cycle=True)
        assert decomposer is not None

    def test_8_observations_minimum_without_cycle_accepted(self):
        """Exactly 8 observations with cycle=False should be accepted."""
        series = _make_series(n=8)
        decomposer = InflationDecomposer(series=series, cycle=False)
        assert decomposer is not None


# ---------------------------------------------------------------------------
# Fit tests — result type and structure
# ---------------------------------------------------------------------------

class TestInflationDecomposerFit:
    def test_returns_correct_type(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert isinstance(result, InflationDecompositionResult)

    def test_all_components_are_polars_series(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        for attr in ("trend", "cycle", "seasonal", "irregular"):
            assert isinstance(getattr(result, attr), pl.Series), f"{attr} is not a Polars Series"

    def test_component_lengths_match_n_obs(self, clean_series, clean_periods):
        n = len(clean_series)
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        for attr in ("trend", "cycle", "seasonal", "irregular"):
            assert len(getattr(result, attr)) == n, f"{attr} has wrong length"

    def test_periods_series_length_matches(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert len(result.periods) == len(clean_series)

    def test_n_obs_matches_input(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert result.n_obs == len(clean_series)

    def test_structural_rate_is_float(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert isinstance(result.structural_rate, float)

    def test_structural_rate_positive_for_growing_series(self):
        """A series with clear 7% pa structural growth should show positive structural rate."""
        series = _make_series(n=40, structural_pa=0.07, cycle_amplitude=0.02, seed=7)
        result = InflationDecomposer(series=series, cycle=True).fit()
        assert result.structural_rate > 0.0, (
            f"structural_rate={result.structural_rate:.4f} should be positive"
        )

    def test_structural_rate_negative_for_falling_series(self):
        """A series with negative structural trend should show negative structural rate."""
        series = _make_series(n=40, structural_pa=-0.04, cycle_amplitude=0.01, seed=8)
        result = InflationDecomposer(series=series, cycle=True).fit()
        assert result.structural_rate < 0.0, (
            f"structural_rate={result.structural_rate:.4f} should be negative"
        )

    def test_total_trend_rate_is_float(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert isinstance(result.total_trend_rate, float)

    def test_total_trend_rate_positive_for_growing_series(self):
        series = _make_series(n=40, structural_pa=0.07, cycle_amplitude=0.02, seed=9)
        result = InflationDecomposer(series=series, cycle=True).fit()
        assert result.total_trend_rate > 0.0

    def test_aic_is_float(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert isinstance(result.aic, float)

    def test_bic_is_float(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert isinstance(result.bic, float)

    def test_bic_ge_aic_for_reasonable_data(self, clean_series, clean_periods):
        """BIC penalises parameters more heavily than AIC, so BIC >= AIC when n is large enough."""
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        # BIC >= AIC when n >= e^2 ≈ 7.4, which is always true here
        assert result.bic >= result.aic, (
            f"BIC={result.bic:.1f} < AIC={result.aic:.1f}"
        )

    def test_log_transform_attribute_preserved(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods, log_transform=True
        ).fit()
        assert result.log_transform is True

    def test_periods_per_year_attribute_preserved(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods, periods_per_year=4
        ).fit()
        assert result.periods_per_year == 4

    def test_cycle_period_years_in_reasonable_range(self):
        """For a 40-period quarterly series with 6-year cycle, estimated period should be roughly in bounds."""
        series = _make_series(n=40, cycle_period_years=6.0, seed=15)
        result = InflationDecomposer(
            series=series,
            cycle=True,
            cycle_period_bounds=(2.0, 12.0),
        ).fit()
        # The period should be within the supplied bounds
        assert 2.0 <= result.cycle_period <= 12.0, (
            f"Estimated cycle period {result.cycle_period:.2f} yrs outside (2, 12) yr bounds"
        )

    def test_cyclical_position_is_finite_float(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert math.isfinite(result.cyclical_position)

    def test_no_cycle_model_runs(self):
        """cycle=False should produce a result with zero cycle component."""
        series = _make_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert isinstance(result, InflationDecompositionResult)
        # With cycle=False, cycle component should be all zeros
        assert np.allclose(result.cycle.to_numpy(), 0.0)

    def test_no_cycle_cycle_period_is_nan(self):
        series = _make_series(n=24)
        result = InflationDecomposer(series=series, cycle=False).fit()
        assert math.isnan(result.cycle_period)


# ---------------------------------------------------------------------------
# Seasonal component tests
# ---------------------------------------------------------------------------

class TestSeasonalComponent:
    def test_with_seasonal_4_runs(self):
        """seasonal=4 on quarterly data should produce a non-trivial seasonal component."""
        rng = np.random.default_rng(20)
        n = 40
        t = np.arange(n, dtype=float)
        # Add explicit quarterly seasonal pattern
        seasonal_effect = 0.05 * np.sin(2 * np.pi * t / 4)
        base = 100.0 * np.exp(0.07 / 4 * t + seasonal_effect)
        noise = rng.normal(0, 0.01, n)
        series = base * np.exp(noise)
        result = InflationDecomposer(
            series=series, seasonal=4, cycle=True
        ).fit()
        assert isinstance(result, InflationDecompositionResult)
        assert len(result.seasonal) == n

    def test_without_seasonal_component_is_zero(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods, seasonal=None
        ).fit()
        assert np.allclose(result.seasonal.to_numpy(), 0.0), (
            "seasonal component should be zero when seasonal=None"
        )


# ---------------------------------------------------------------------------
# Component reconstruction tests
# ---------------------------------------------------------------------------

class TestComponentReconstruction:
    def test_components_sum_to_observed_approximately(self):
        """trend + cycle + seasonal + irregular should approximately equal observed."""
        series = _make_series(n=40, seed=30)
        result = InflationDecomposer(series=series).fit()

        obs = result.observations.to_numpy()
        reconstructed = (
            result.trend.to_numpy()
            + result.cycle.to_numpy()
            + result.seasonal.to_numpy()
            + result.irregular.to_numpy()
        )
        # Allow generous tolerance — the Kalman smoother attribution involves
        # diffuse initialisation adjustments that can shift the first few obs.
        max_deviation = np.max(np.abs(reconstructed - obs))
        assert max_deviation < 0.5, (
            f"Max component reconstruction error {max_deviation:.4f} > 0.5"
        )

    def test_observations_are_log_transformed_when_flag_true(self):
        """observations should be on the log scale when log_transform=True."""
        series = _make_series(n=40, base=100.0, seed=31)
        result = InflationDecomposer(series=series, log_transform=True).fit()
        # Log-transformed: all observations should be positive (since log(100+) > 0)
        # and much smaller than original scale
        obs = result.observations.to_numpy()
        # log(100) ≈ 4.6, so all values should be around 4-6 for this series
        assert np.all(obs > 3.0), "Log-transformed observations should be around log(100)"
        assert np.all(obs < 10.0), "Log-transformed observations should be small"

    def test_observations_untransformed_when_flag_false(self):
        """When log_transform=False, observations should be the raw values."""
        series = _make_series(n=40, base=100.0, seed=32)
        result = InflationDecomposer(
            series=series, log_transform=False, cycle=False
        ).fit()
        obs = result.observations.to_numpy()
        # Should be on original scale (around 100)
        assert np.all(obs > 50.0), "Untransformed observations should be on original scale"


# ---------------------------------------------------------------------------
# Result helper methods
# ---------------------------------------------------------------------------

class TestInflationDecompositionResultMethods:
    def test_summary_returns_string(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_summary_contains_structural_rate(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert "Structural trend" in result.summary()

    def test_summary_contains_cycle_period(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert "period" in result.summary().lower()

    def test_summary_contains_aic(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert "AIC" in result.summary()

    def test_decomposition_table_returns_dataframe(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        table = result.decomposition_table()
        assert isinstance(table, pl.DataFrame)

    def test_decomposition_table_has_expected_columns(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        table = result.decomposition_table()
        expected_cols = {"period", "observed", "trend", "cycle", "seasonal", "irregular"}
        assert set(table.columns) == expected_cols

    def test_decomposition_table_row_count(self, clean_series, clean_periods):
        n = len(clean_series)
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert len(result.decomposition_table()) == n

    def test_repr_contains_n_obs(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert "n_obs=40" in repr(result)

    def test_repr_contains_structural_rate(self, clean_series, clean_periods):
        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        assert "structural_rate" in repr(result)

    def test_plot_returns_figure(self, clean_series, clean_periods):
        """plot() should return a matplotlib Figure without raising."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for tests

        result = InflationDecomposer(
            series=clean_series, periods=clean_periods
        ).fit()
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Monthly data tests
# ---------------------------------------------------------------------------

class TestMonthlyData:
    def test_monthly_data_periods_per_year_12(self):
        """Monthly series with periods_per_year=12 should run without error."""
        rng = np.random.default_rng(50)
        n = 60  # 5 years of monthly data
        t = np.arange(n, dtype=float)
        series = 100.0 * np.exp(0.06 / 12 * t + rng.normal(0, 0.01, n))
        result = InflationDecomposer(
            series=series,
            periods_per_year=12,
            cycle=True,
            cycle_period_bounds=(2.0, 8.0),
        ).fit()
        assert isinstance(result, InflationDecompositionResult)
        assert result.n_obs == n
        assert result.periods_per_year == 12

    def test_monthly_structural_rate_annualised(self):
        """A 6% pa structural trend in monthly data should be recovered approximately."""
        rng = np.random.default_rng(51)
        n = 72  # 6 years of monthly data
        t = np.arange(n, dtype=float)
        series = 100.0 * np.exp(0.06 / 12 * t + rng.normal(0, 0.005, n))
        result = InflationDecomposer(
            series=series,
            periods_per_year=12,
            cycle=True,
            cycle_period_bounds=(2.0, 8.0),
        ).fit()
        # Should be in the right ballpark (1% to 15%) for a series trending at 6% pa
        assert 0.01 < result.structural_rate < 0.20, (
            f"structural_rate={result.structural_rate:.4f} out of expected range for 6% pa series"
        )


# ---------------------------------------------------------------------------
# Input type robustness
# ---------------------------------------------------------------------------

class TestInputTypeRobustness:
    def test_numpy_array_input(self):
        series = np.array(_make_series(n=24))
        result = InflationDecomposer(series=series).fit()
        assert isinstance(result, InflationDecompositionResult)

    def test_list_input(self):
        series = list(_make_series(n=24))
        result = InflationDecomposer(series=series).fit()
        assert isinstance(result, InflationDecompositionResult)

    def test_polars_series_input(self):
        series = pl.Series("sev", _make_series(n=24))
        result = InflationDecomposer(series=series).fit()
        assert isinstance(result, InflationDecompositionResult)

    def test_pandas_series_input(self):
        series = pd.Series(_make_series(n=24))
        result = InflationDecomposer(series=series).fit()
        assert isinstance(result, InflationDecompositionResult)


# ---------------------------------------------------------------------------
# Export and __init__ integration
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_inflation_decomposer_importable_from_package(self):
        from insurance_trend import InflationDecomposer as ID
        assert ID is not None

    def test_inflation_result_importable_from_package(self):
        from insurance_trend import InflationDecompositionResult as IDR
        assert IDR is not None

    def test_inflation_decomposer_in_all(self):
        import insurance_trend
        assert "InflationDecomposer" in insurance_trend.__all__

    def test_inflation_result_in_all(self):
        import insurance_trend
        assert "InflationDecompositionResult" in insurance_trend.__all__
