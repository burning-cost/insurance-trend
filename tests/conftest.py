"""Shared fixtures for insurance-trend tests."""

import numpy as np
import pytest
import polars as pl


@pytest.fixture
def quarterly_periods():
    """16 quarters of period labels."""
    return [
        f"{year}Q{q}"
        for year in range(2020, 2024)
        for q in range(1, 5)
    ]


@pytest.fixture
def flat_trend_data():
    """Synthetic data with no trend (flat frequency and severity).

    Returns a dict with keys: periods, claim_counts, earned_exposure, total_paid.
    """
    rng = np.random.default_rng(0)
    n = 16
    exposure = np.full(n, 1000.0) + rng.normal(0, 10, n)
    freq = 0.10  # 10 claims per 100 exposure
    claim_counts = np.round(exposure * freq).astype(float)
    severity = 5000.0  # £5,000 average severity
    total_paid = claim_counts * severity + rng.normal(0, 5000, n)
    periods = [f"{year}Q{q}" for year in range(2020, 2024) for q in range(1, 5)]
    return {
        "periods": periods,
        "claim_counts": claim_counts,
        "earned_exposure": exposure,
        "total_paid": total_paid,
    }


@pytest.fixture
def trending_data():
    """Synthetic data with a genuine positive trend.

    Frequency trend: -2% pa (negative frequency, positive severity driven)
    Severity trend: +8% pa
    Combined: approximately +5.8% pa
    """
    rng = np.random.default_rng(1)
    n = 20
    t = np.arange(n)
    exposure = np.full(n, 1000.0)
    freq_pa = -0.02
    sev_pa = 0.08
    per_period_freq = (1 + freq_pa) ** (1 / 4) - 1
    per_period_sev = (1 + sev_pa) ** (1 / 4) - 1

    base_freq = 0.12
    base_sev = 4800.0

    frequency = base_freq * (1 + per_period_freq) ** t * np.exp(rng.normal(0, 0.02, n))
    severity = base_sev * (1 + per_period_sev) ** t * np.exp(rng.normal(0, 0.03, n))

    claim_counts = np.maximum(1.0, np.round(exposure * frequency))
    total_paid = claim_counts * severity

    periods = [f"{year}Q{q}" for year in range(2019, 2024) for q in range(1, 5)][:n]
    return {
        "periods": periods,
        "claim_counts": claim_counts,
        "earned_exposure": exposure,
        "total_paid": total_paid,
        "expected_freq_trend": freq_pa,
        "expected_sev_trend": sev_pa,
    }


@pytest.fixture
def polars_trending_data(trending_data):
    """Same as trending_data but as Polars Series."""
    return {
        "periods": pl.Series("periods", trending_data["periods"]),
        "claim_counts": pl.Series("claim_counts", trending_data["claim_counts"]),
        "earned_exposure": pl.Series("earned_exposure", trending_data["earned_exposure"]),
        "total_paid": pl.Series("total_paid", trending_data["total_paid"]),
        "expected_freq_trend": trending_data["expected_freq_trend"],
        "expected_sev_trend": trending_data["expected_sev_trend"],
    }


@pytest.fixture
def breakpoint_data():
    """Synthetic data with a structural break at index 8 (simulating COVID).

    Pre-break: frequency 0.12, severity 4800
    Post-break: frequency 0.07, severity 5800 (step change)
    """
    rng = np.random.default_rng(2)
    n = 20
    exposure = np.full(n, 1000.0)
    freq = np.concatenate([
        np.full(8, 0.12) * np.exp(rng.normal(0, 0.01, 8)),
        np.full(12, 0.07) * np.exp(rng.normal(0, 0.01, 12)),
    ])
    sev = np.concatenate([
        np.full(8, 4800.0) * np.exp(rng.normal(0, 0.02, 8)),
        np.full(12, 5800.0) * np.exp(rng.normal(0, 0.02, 12)),
    ])
    claim_counts = np.maximum(1.0, np.round(exposure * freq))
    total_paid = claim_counts * sev

    periods = [f"{year}Q{q}" for year in range(2019, 2024) for q in range(1, 5)][:n]
    return {
        "periods": periods,
        "claim_counts": claim_counts,
        "earned_exposure": exposure,
        "total_paid": total_paid,
        "break_index": 8,
    }


@pytest.fixture
def external_index_series():
    """Synthetic external index representing ~4% pa inflation over 20 quarters."""
    rng = np.random.default_rng(3)
    n = 20
    t = np.arange(n)
    index = 100.0 * np.exp(0.04 / 4 * t) * np.exp(rng.normal(0, 0.005, n))
    return pl.Series("HPTH", index)


@pytest.fixture
def short_data():
    """Minimal valid dataset: 6 quarters."""
    rng = np.random.default_rng(4)
    n = 6
    exposure = np.full(n, 800.0)
    counts = np.round(exposure * 0.10).astype(float)
    paid = counts * 4500.0 + rng.normal(0, 1000, n)
    periods = [f"2022Q{q}" for q in range(1, 5)] + ["2023Q1", "2023Q2"]
    return {
        "periods": periods,
        "claim_counts": counts,
        "earned_exposure": exposure,
        "total_paid": paid,
    }
