"""Tests for ExternalIndex — ONS API fetcher and CSV loader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import pytest

from insurance_trend import ExternalIndex


# ------------------------------------------------------------------
# Mock ONS API response fixture
# ------------------------------------------------------------------

def _make_ons_response(code: str = "HPTH", n_quarters: int = 24) -> dict:
    """Build a minimal ONS-style JSON response for testing."""
    quarters = []
    year = 2018
    q = 1
    for i in range(n_quarters):
        date_str = f"{year} Q{q}"
        quarters.append({"date": f"{year}-Q{q}", "value": str(100.0 + i * 0.8)})
        q += 1
        if q > 4:
            q = 1
            year += 1
    return {
        "description": {"title": f"Test series {code}"},
        "quarters": quarters,
    }


class TestExternalIndexCatalogue:
    def test_catalogue_is_dict(self):
        assert isinstance(ExternalIndex.CATALOGUE, dict)

    def test_catalogue_contains_hpth(self):
        assert "motor_repair" in ExternalIndex.CATALOGUE
        assert ExternalIndex.CATALOGUE["motor_repair"] == "HPTH"

    def test_catalogue_contains_l7je(self):
        assert "motor_insurance_cpi" in ExternalIndex.CATALOGUE

    def test_catalogue_contains_d7do(self):
        assert "building_maintenance" in ExternalIndex.CATALOGUE

    def test_list_catalogue_returns_dataframe(self):
        df = ExternalIndex.list_catalogue()
        assert isinstance(df, pl.DataFrame)
        assert "name" in df.columns
        assert "ons_code" in df.columns

    def test_list_catalogue_non_empty(self):
        df = ExternalIndex.list_catalogue()
        assert len(df) > 0


class TestExternalIndexFromSeries:
    def test_from_numpy_array(self):
        arr = np.array([100.0, 101.5, 103.2, 105.0])
        s = ExternalIndex.from_series(arr, label="test")
        assert isinstance(s, pl.Series)
        assert len(s) == 4

    def test_from_polars_series(self):
        ps = pl.Series("my_index", [100.0, 102.0, 104.0])
        s = ExternalIndex.from_series(ps, label="renamed")
        assert isinstance(s, pl.Series)
        assert s.name == "renamed"

    def test_from_pandas_series(self):
        ps = pd.Series([100.0, 102.0, 104.0], name="pandas_idx")
        s = ExternalIndex.from_series(ps, label="converted")
        assert isinstance(s, pl.Series)
        assert len(s) == 3

    def test_from_list(self):
        s = ExternalIndex.from_series([100.0, 101.0, 102.0], label="list_idx")
        assert isinstance(s, pl.Series)

    def test_values_preserved(self):
        arr = np.array([95.0, 100.0, 105.0, 110.0])
        s = ExternalIndex.from_series(arr)
        np.testing.assert_allclose(s.to_numpy(), arr)


class TestExternalIndexFromCsv:
    def test_basic_csv_load(self, tmp_path):
        csv_path = tmp_path / "test_index.csv"
        csv_path.write_text("date,index_value\n2020-01-01,100.0\n2020-04-01,101.5\n2020-07-01,103.0\n")
        s = ExternalIndex.from_csv(str(csv_path), date_col="date", value_col="index_value")
        assert isinstance(s, pl.Series)
        assert len(s) == 3

    def test_values_loaded_correctly(self, tmp_path):
        csv_path = tmp_path / "idx.csv"
        csv_path.write_text("period,val\n2020Q1,100.0\n2020Q2,102.0\n2020Q3,104.0\n")
        s = ExternalIndex.from_csv(str(csv_path), date_col="period", value_col="val")
        np.testing.assert_allclose(s.to_numpy(), [100.0, 102.0, 104.0])

    def test_missing_date_col_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")
        with pytest.raises(ValueError, match="date_col"):
            ExternalIndex.from_csv(str(csv_path), date_col="date", value_col="y")

    def test_missing_value_col_raises(self, tmp_path):
        csv_path = tmp_path / "bad2.csv"
        csv_path.write_text("date,wrong\n2020Q1,100\n")
        with pytest.raises(ValueError, match="value_col"):
            ExternalIndex.from_csv(str(csv_path), date_col="date", value_col="index_value")

    def test_result_is_float64(self, tmp_path):
        csv_path = tmp_path / "int_vals.csv"
        csv_path.write_text("d,v\n2020Q1,100\n2020Q2,102\n")
        s = ExternalIndex.from_csv(str(csv_path), date_col="d", value_col="v")
        assert s.dtype == pl.Float64


class TestONSResponseParsing:
    """Test the internal _parse_ons_response method without network calls."""

    def test_parses_quarters(self):
        data = _make_ons_response("HPTH", 16)
        s = ExternalIndex._parse_ons_response(data, "HPTH", "quarters", "2018-01-01")
        assert isinstance(s, pl.Series)
        assert len(s) == 16

    def test_filters_by_start_date(self):
        data = _make_ons_response("HPTH", 24)
        # First entry is "2018-Q1"; filtering from 2020 should exclude ~8 quarters
        s = ExternalIndex._parse_ons_response(data, "HPTH", "quarters", "2020-01-01")
        assert len(s) < 24

    def test_falls_back_to_months_if_quarters_missing(self):
        data = {
            "months": [{"date": "2020-01", "value": "100.0"}] * 12,
        }
        with pytest.warns(UserWarning, match="only 'months' available"):
            s = ExternalIndex._parse_ons_response(data, "TEST", "quarters", "2015-01-01")
        assert len(s) == 12

    def test_raises_if_no_valid_frequency(self):
        data = {"years": [{"date": "2020", "value": "100.0"}]}
        with pytest.raises(ValueError, match="no 'quarters' data"):
            ExternalIndex._parse_ons_response(data, "TEST", "quarters", "2015-01-01")

    def test_skips_non_numeric_values(self):
        data = {
            "quarters": [
                {"date": "2020-Q1", "value": "100.0"},
                {"date": "2020-Q2", "value": "N/A"},  # non-numeric
                {"date": "2020-Q3", "value": "102.0"},
            ]
        }
        s = ExternalIndex._parse_ons_response(data, "X", "quarters", "2015-01-01")
        assert len(s) == 2  # N/A entry skipped

    def test_series_named_by_code(self):
        data = _make_ons_response("HPTH", 8)
        s = ExternalIndex._parse_ons_response(data, "HPTH", "quarters", "2015-01-01")
        assert s.name == "HPTH"

    def test_raises_for_empty_entries_after_filter(self):
        data = {
            "quarters": [
                {"date": "2010-Q1", "value": "100.0"},
                {"date": "2010-Q2", "value": "101.0"},
            ]
        }
        with pytest.raises(ValueError, match="no numeric values found"):
            ExternalIndex._parse_ons_response(data, "X", "quarters", "2025-01-01")


class TestFromOnsCachePath:
    """Test the cache_path argument with mocked HTTP."""

    def test_cache_written_and_read(self, tmp_path, monkeypatch):
        """ONS response should be written to cache on first call; read on second."""
        import requests

        call_count = {"n": 0}
        data = _make_ons_response("HPTH", 16)

        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                call_count["n"] += 1
                return data

        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())

        cache = str(tmp_path / "hpth_cache.json")

        # First call: should hit network (via monkeypatched requests.get)
        s1 = ExternalIndex.from_ons("HPTH", cache_path=cache)
        assert call_count["n"] == 1

        # Second call: should read from cache, not hit network
        s2 = ExternalIndex.from_ons("HPTH", cache_path=cache)
        assert call_count["n"] == 1  # No additional call

        np.testing.assert_allclose(s1.to_numpy(), s2.to_numpy())

    def test_http_error_raises(self, monkeypatch):
        import requests

        class BadResponse:
            def raise_for_status(self):
                raise requests.HTTPError("404 Not Found")
            def json(self):
                return {}

        monkeypatch.setattr(requests, "get", lambda *a, **kw: BadResponse())

        with pytest.raises(requests.HTTPError):
            ExternalIndex.from_ons("XXXX")
