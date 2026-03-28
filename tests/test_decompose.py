"""Tests for MultiIndexDecomposer and MultiIndexResult."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pandas as pd
import pytest

from insurance_trend import MultiIndexDecomposer, MultiIndexResult
from insurance_trend.decompose import UK_INSURANCE_EVENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_periods(n: int) -> list[str]:
    """Generate n quarterly period labels starting 2019Q1."""
    labels = []
    for year in range(2019, 2035):
        for q in range(1, 5):
            labels.append(f"{year}Q{q}")
            if len(labels) == n:
                return labels
    return labels


def _two_index_synthetic(
    n: int = 20,
    beta_parts: float = 0.6,
    beta_labour: float = 0.4,
    parts_pa: float = 0.06,
    labour_pa: float = 0.04,
    noise_sigma: float = 0.005,
    seed: int = 42,
) -> dict:
    """Synthetic severity driven by two indices with known elasticities.

    severity = base * parts^beta_parts * labour^beta_labour * noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    parts = np.exp(parts_pa / 4 * t)
    labour = np.exp(labour_pa / 4 * t)
    severity = 5000.0 * (parts ** beta_parts) * (labour ** beta_labour)
    severity *= np.exp(rng.normal(0, noise_sigma, n))
    return {
        "periods": _make_periods(n),
        "severity": severity,
        "parts": parts,
        "labour": labour,
        "beta_parts": beta_parts,
        "beta_labour": beta_labour,
        "parts_pa": parts_pa,
        "labour_pa": labour_pa,
    }


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestMultiIndexDecomposerConstruction:
    def test_basic_construction(self):
        d = _two_index_synthetic()
        decomposer = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        )
        assert decomposer is not None

    def test_polars_series_input(self):
        d = _two_index_synthetic()
        decomposer = MultiIndexDecomposer(
            periods=pl.Series("p", d["periods"]),
            severity=pl.Series("sev", d["severity"]),
            indices={
                "parts": pl.Series("parts", d["parts"]),
                "labour": pl.Series("labour", d["labour"]),
            },
        )
        result = decomposer.fit()
        assert isinstance(result, MultiIndexResult)

    def test_pandas_series_input(self):
        d = _two_index_synthetic()
        decomposer = MultiIndexDecomposer(
            periods=pd.Series(d["periods"]),
            severity=pd.Series(d["severity"]),
            indices={
                "parts": pd.Series(d["parts"]),
                "labour": pd.Series(d["labour"]),
            },
        )
        result = decomposer.fit()
        assert isinstance(result, MultiIndexResult)

    def test_list_input(self):
        d = _two_index_synthetic()
        decomposer = MultiIndexDecomposer(
            periods=list(d["periods"]),
            severity=list(d["severity"]),
            indices={"parts": list(d["parts"])},
        )
        result = decomposer.fit()
        assert isinstance(result, MultiIndexResult)

    def test_empty_indices_raises(self):
        d = _two_index_synthetic()
        with pytest.raises(ValueError, match="at least one"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={},
            )

    def test_zero_severity_raises(self):
        d = _two_index_synthetic()
        bad_sev = d["severity"].copy()
        bad_sev[3] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=bad_sev,
                indices={"parts": d["parts"]},
            )

    def test_negative_severity_raises(self):
        d = _two_index_synthetic()
        bad_sev = d["severity"].copy()
        bad_sev[0] = -100.0
        with pytest.raises(ValueError, match="strictly positive"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=bad_sev,
                indices={"parts": d["parts"]},
            )

    def test_mismatched_index_length_raises(self):
        d = _two_index_synthetic(n=16)
        with pytest.raises(ValueError, match="same length"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={"parts": d["parts"][:-3]},  # too short
            )

    def test_mismatched_weights_length_raises(self):
        d = _two_index_synthetic()
        with pytest.raises(ValueError, match="weights length"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={"parts": d["parts"]},
                weights=np.ones(5),  # wrong length
            )

    def test_non_positive_index_raises(self):
        d = _two_index_synthetic()
        bad_idx = d["parts"].copy()
        bad_idx[2] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={"parts": bad_idx},
            )

    def test_invalid_periods_per_year_raises(self):
        d = _two_index_synthetic()
        with pytest.raises(ValueError, match="periods_per_year"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={"parts": d["parts"]},
                periods_per_year=0,
            )

    def test_negative_periods_per_year_raises(self):
        d = _two_index_synthetic()
        with pytest.raises(ValueError, match="periods_per_year"):
            MultiIndexDecomposer(
                periods=d["periods"],
                severity=d["severity"],
                indices={"parts": d["parts"]},
                periods_per_year=-4,
            )


# ---------------------------------------------------------------------------
# Fit tests — coefficient recovery
# ---------------------------------------------------------------------------

class TestMultiIndexDecomposerFit:
    def test_returns_multi_index_result(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert isinstance(result, MultiIndexResult)

    def test_coefficients_dict_keys(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert set(result.coefficients.keys()) == {"parts", "labour"}

    def test_annual_contributions_dict_keys(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert set(result.annual_contributions.keys()) == {"parts", "labour"}

    def test_coefficients_recovered_approximately(self):
        """OLS should recover elasticities close to the true values (low-noise case)."""
        d = _two_index_synthetic(n=40, noise_sigma=0.002, seed=0)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert abs(result.coefficients["parts"] - d["beta_parts"]) < 0.15, (
            f"parts coeff {result.coefficients['parts']:.4f} far from true {d['beta_parts']}"
        )
        assert abs(result.coefficients["labour"] - d["beta_labour"]) < 0.15, (
            f"labour coeff {result.coefficients['labour']:.4f} far from true {d['beta_labour']}"
        )

    def test_r_squared_high_for_clean_data(self):
        """Near-noiseless data should produce R² close to 1."""
        d = _two_index_synthetic(n=40, noise_sigma=0.001, seed=1)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert result.r_squared > 0.95, f"R² = {result.r_squared:.4f}, expected > 0.95"

    def test_r_squared_in_valid_range(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert 0.0 <= result.r_squared <= 1.0

    def test_residual_near_zero_for_fully_explained_data(self):
        """When severity is exactly driven by the indices, residual should be ~0."""
        d = _two_index_synthetic(n=40, noise_sigma=0.001, seed=2)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert abs(result.residual_rate) < 0.03, (
            f"Residual {result.residual_rate:.4f} too large for near-noiseless data"
        )

    def test_total_severity_trend_sign(self):
        """When indices grow, total severity trend should be positive."""
        d = _two_index_synthetic(n=20, parts_pa=0.06, labour_pa=0.04)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert result.total_severity_trend > 0.0

    def test_single_index_fit(self):
        """Single-index case should work and return a single-key result."""
        rng = np.random.default_rng(10)
        n = 20
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 4 * t)
        severity = 3000.0 * (idx ** 0.8) * np.exp(rng.normal(0, 0.01, n))
        result = MultiIndexDecomposer(
            periods=_make_periods(n),
            severity=severity,
            indices={"motor_repair": idx},
        ).fit()
        assert len(result.coefficients) == 1
        assert "motor_repair" in result.coefficients

    def test_monthly_data_periods_per_year_12(self):
        """periods_per_year=12 should not crash and should return plausible results."""
        rng = np.random.default_rng(11)
        n = 48
        t = np.arange(n, dtype=float)
        idx = np.exp(0.05 / 12 * t)
        severity = 2000.0 * (idx ** 0.7) * np.exp(rng.normal(0, 0.01, n))
        result = MultiIndexDecomposer(
            periods=list(range(n)),
            severity=severity,
            indices={"cpi": idx},
            periods_per_year=12,
        ).fit()
        assert isinstance(result, MultiIndexResult)
        assert result.r_squared >= 0.0


# ---------------------------------------------------------------------------
# Decomposition table tests
# ---------------------------------------------------------------------------

class TestDecompositionTable:
    def test_table_is_polars_dataframe(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert isinstance(result.decomposition_table, pl.DataFrame)

    def test_table_columns(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        expected_cols = {
            "component",
            "coefficient",
            "annual_contribution_pct",
            "share_of_total_pct",
        }
        assert set(result.decomposition_table.columns) == expected_cols

    def test_table_row_count(self):
        """Should have n_indices + 1 rows (one Residual row)."""
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert len(result.decomposition_table) == 3  # 2 indices + Residual

    def test_table_has_residual_row(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        components = result.decomposition_table["component"].to_list()
        assert "Residual" in components

    def test_shares_sum_to_100_for_clean_data(self):
        """For clean data with no residual, shares should sum to approximately 100%."""
        d = _two_index_synthetic(n=40, noise_sigma=0.0005, seed=3)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        total_share = result.decomposition_table["share_of_total_pct"].sum()
        assert abs(total_share - 100.0) < 5.0, (
            f"Shares sum to {total_share:.2f}%, expected ~100%"
        )

    def test_contributions_sum_approximately_equals_total_trend(self):
        """Index contributions + residual should approximately equal total trend."""
        d = _two_index_synthetic(n=40, noise_sigma=0.001, seed=4)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        # Sum of annual_contribution_pct / 100 + residual_rate ≈ total_severity_trend
        index_sum_pa = sum(result.annual_contributions.values())
        reconstructed = index_sum_pa + result.residual_rate
        assert abs(reconstructed - result.total_severity_trend) < 0.01, (
            f"Reconstructed {reconstructed:.4f} != total {result.total_severity_trend:.4f}"
        )


# ---------------------------------------------------------------------------
# Weighted fit tests
# ---------------------------------------------------------------------------

class TestWeightedFit:
    def test_weights_accepted(self):
        d = _two_index_synthetic()
        n = len(d["severity"])
        weights = np.linspace(0.5, 1.0, n)
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
            weights=weights,
        ).fit()
        assert isinstance(result, MultiIndexResult)

    def test_uniform_weights_same_as_no_weights(self):
        """Uniform weights must produce the same result as OLS."""
        d = _two_index_synthetic(seed=50)
        n = len(d["severity"])
        result_ols = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"]},
        ).fit()
        result_wls = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"]},
            weights=np.ones(n),
        ).fit()
        assert abs(result_ols.coefficients["parts"] - result_wls.coefficients["parts"]) < 1e-6

    def test_downweighted_outlier_changes_coefficients(self):
        """Downweighting a severely distorted period should shift coefficients."""
        rng = np.random.default_rng(99)
        d = _two_index_synthetic(n=20, noise_sigma=0.001, seed=99)
        sev_with_spike = d["severity"].copy()
        sev_with_spike[5] *= 5.0  # large outlier at period 5

        weights_equal = np.ones(20)
        weights_downweighted = np.ones(20)
        weights_downweighted[5] = 0.01

        result_equal = MultiIndexDecomposer(
            periods=d["periods"],
            severity=sev_with_spike,
            indices={"parts": d["parts"]},
            weights=weights_equal,
        ).fit()
        result_down = MultiIndexDecomposer(
            periods=d["periods"],
            severity=sev_with_spike,
            indices={"parts": d["parts"]},
            weights=weights_downweighted,
        ).fit()
        # The two fits should differ meaningfully
        diff = abs(result_equal.coefficients["parts"] - result_down.coefficients["parts"])
        assert diff > 0.01, (
            f"Downweighted outlier did not change fit: diff={diff:.4f}"
        )


# ---------------------------------------------------------------------------
# MultiIndexResult methods
# ---------------------------------------------------------------------------

class TestMultiIndexResult:
    def test_summary_returns_string(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_total_trend(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        s = result.summary()
        assert "Total severity trend" in s

    def test_summary_contains_residual(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        s = result.summary()
        assert "Residual" in s

    def test_repr_contains_n_indices(self):
        d = _two_index_synthetic()
        result = MultiIndexDecomposer(
            periods=d["periods"],
            severity=d["severity"],
            indices={"parts": d["parts"], "labour": d["labour"]},
        ).fit()
        assert "n_indices=2" in repr(result)

    def test_three_indices(self):
        """Three-index decomposition should have three coefficient keys."""
        rng = np.random.default_rng(7)
        n = 24
        t = np.arange(n, dtype=float)
        parts = np.exp(0.06 / 4 * t)
        labour = np.exp(0.04 / 4 * t)
        legal = np.exp(0.03 / 4 * t)
        severity = 4000.0 * (parts ** 0.5) * (labour ** 0.3) * (legal ** 0.2)
        severity *= np.exp(rng.normal(0, 0.005, n))
        result = MultiIndexDecomposer(
            periods=_make_periods(n),
            severity=severity,
            indices={"parts": parts, "labour": labour, "legal": legal},
        ).fit()
        assert len(result.coefficients) == 3
        assert len(result.decomposition_table) == 4  # 3 indices + Residual


# ---------------------------------------------------------------------------
# UK insurance events dict
# ---------------------------------------------------------------------------

class TestUKInsuranceEvents:
    def test_is_dict(self):
        assert isinstance(UK_INSURANCE_EVENTS, dict)

    def test_non_empty(self):
        assert len(UK_INSURANCE_EVENTS) > 0

    def test_keys_are_strings(self):
        for k in UK_INSURANCE_EVENTS:
            assert isinstance(k, str)

    def test_values_are_strings(self):
        for v in UK_INSURANCE_EVENTS.values():
            assert isinstance(v, str)

    def test_covid_event_present(self):
        assert any("COVID" in v or "covid" in v for v in UK_INSURANCE_EVENTS.values())


# ---------------------------------------------------------------------------
# Index catalogue additions
# ---------------------------------------------------------------------------

class TestIndexCatalogueAdditions:
    def test_new_codes_present(self):
        from insurance_trend import ExternalIndex
        catalogue = ExternalIndex.CATALOGUE
        for code in ("HPTD", "KAB9", "KAC3", "L522", "L7GA"):
            assert code in catalogue.values(), f"ONS code {code} missing from catalogue"

    def test_sppi_url_routing(self):
        from insurance_trend.index import _ons_url_for
        hpth_url = _ons_url_for("HPTH")
        hptd_url = _ons_url_for("HPTD")
        assert "/sppi/" in hpth_url, f"HPTH should route to /sppi/, got: {hpth_url}"
        assert "/sppi/" in hptd_url, f"HPTD should route to /sppi/, got: {hptd_url}"

    def test_mm23_url_routing_for_cpi(self):
        from insurance_trend.index import _ons_url_for
        cpi_url = _ons_url_for("L7JE")
        assert "/mm23/" in cpi_url, f"L7JE should route to /mm23/, got: {cpi_url}"
