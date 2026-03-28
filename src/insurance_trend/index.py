"""ExternalIndex — loader for external inflation indices.

Supports:
- ONS public API (no authentication required)
- User-supplied CSV files (for BCIS and other subscription data)
- Direct Series input

The ONS API returns data at distinct endpoints depending on the publication:

    MM23 (CPI/RPI):
        https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{CODE}/mm23/data

    SPPI (Services Producer Price Index):
        https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{CODE}/sppi/data

    AWE (Average Weekly Earnings, labour market):
        https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/{CODE}/emp/data

The class routes each series code to the correct endpoint automatically based on
``_SPPI_CODES`` and ``_AWE_CODES``.

The response contains a ``months`` and/or ``quarters`` array. This module
parses both, converts to a Polars Series indexed by period label, and
optionally caches to avoid repeated network calls.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import requests


# Catalogue of UK insurance-relevant ONS series codes.
# Format: human-readable name -> ONS series code.
_CATALOGUE: dict[str, str] = {
    # SPPI — Services Producer Price Index (motor/property repair)
    "motor_repair": "HPTH",           # SPPI G4520 Maintenance & repair of motor vehicles
    "motor_repair_parts": "HPTD",     # SPPI G4520 Parts component only
    # CPI — Consumer Price Index sub-indices
    "motor_insurance_cpi": "L7JE",    # CPI 12.5.4.1 Motor vehicle insurance
    "used_cars_cpi": "L7JD",          # CPI 07.1.2 Second-hand cars
    "new_cars_cpi": "L7JC",           # CPI 07.1.1 New cars
    "vehicle_fuels_cpi": "L7JF",      # CPI 07.2.2 Fuels and lubricants
    # RPI — Retail Price Index
    "vehicle_maintenance_rpi": "CZEA",# RPI Maintenance of motor vehicles
    # Property/household
    "building_maintenance": "D7DO",   # CPI 04.3.2 Services for maintenance & repair of dwellings
    "household_maintenance_weights": "CJVD",  # CPI Weights 04.3 Maintenance & repair
    # Earnings / labour costs
    "avg_weekly_earnings": "KAB9",    # AWE Total pay, whole economy
    "avg_weekly_earnings_priv": "KAC3",  # AWE Total pay, private sector
    # Wider economy
    "cpi_all_items": "L55O",          # CPI All Items
    "rpi_all_items": "CZBH",          # RPI All Items
    "services_ppi": "L522",           # PPI Output — services
    "ppi_output_all": "L7GA",         # PPI Output — all manufactured products
}

# Series codes that are published under the SPPI endpoint rather than MM23.
# Misrouting these codes to /mm23/data returns a 404 or empty response.
_SPPI_CODES: frozenset[str] = frozenset({"HPTH", "HPTD"})

_ONS_MM23_URL = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{code}/mm23/data"
_ONS_SPPI_URL = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{code}/sppi/data"

# AWE (Average Weekly Earnings) series are published under the labour market /emp/ dataset,
# not the inflation /mm23/ dataset. Routing these to /mm23/ returns an empty response.
_AWE_CODES: frozenset[str] = frozenset({"KAB9", "KAC3"})
_ONS_AWE_URL = (
    "https://www.ons.gov.uk/employmentandlabourmarket"
    "/peopleinwork/earningsandworkinghours"
    "/timeseries/{code}/emp/data"
)

# Default connect/read timeout in seconds for ONS API calls
_DEFAULT_TIMEOUT = 30


def _ons_url_for(code: str) -> str:
    """Return the correct ONS API URL for the given series code.

    Routing:
    - SPPI series (HPTH, HPTD): ``/sppi/data`` endpoint
    - AWE series (KAB9, KAC3): ``/emp/data`` endpoint under the labour market path
    - All others: ``/mm23/data`` endpoint

    Parameters
    ----------
    code:
        ONS series identifier (uppercase).

    Returns
    -------
    str
        Full URL for the series' data endpoint.
    """
    if code in _SPPI_CODES:
        return _ONS_SPPI_URL.format(code=code)
    if code in _AWE_CODES:
        return _ONS_AWE_URL.format(code=code)
    return _ONS_MM23_URL.format(code=code)


class ExternalIndex:
    """Loader for external inflation indices used in severity deflation.

    This class provides class methods for fetching from the ONS public API
    and loading from CSV files. The primary output is a Polars Series whose
    values are the index levels aligned to the periods in your accident data.

    Attributes
    ----------
    CATALOGUE:
        Dict of human-readable name -> ONS series code, covering the most
        commonly used UK insurance indices. Extend as needed.

    Examples
    --------
    >>> from insurance_trend import ExternalIndex
    >>>
    >>> # Fetch ONS motor repair index
    >>> idx = ExternalIndex.from_ons('HPTH')
    >>>
    >>> # Or use the catalogue name
    >>> idx = ExternalIndex.from_ons(ExternalIndex.CATALOGUE['motor_repair'])
    >>>
    >>> # Load BCIS data from CSV
    >>> idx = ExternalIndex.from_csv('bcis_data.csv', date_col='date', value_col='index')
    """

    CATALOGUE: dict[str, str] = _CATALOGUE

    def __init__(self, series: pl.Series, label: str = "index") -> None:
        """Wrap a Polars Series as an ExternalIndex.

        Parameters
        ----------
        series:
            The index values as a Polars Series.
        label:
            Human-readable label for this index.
        """
        self._series = series
        self._label = label

    @property
    def series(self) -> pl.Series:
        """The underlying Polars Series."""
        return self._series

    @property
    def label(self) -> str:
        """Human-readable label for this index."""
        return self._label

    @classmethod
    def from_ons(
        cls,
        series_code: str,
        start_date: str = "2015-01-01",
        frequency: str = "quarters",
        timeout: int = _DEFAULT_TIMEOUT,
        cache_path: Optional[str] = None,
    ) -> pl.Series:
        """Fetch a time series from the ONS public API.

        Returns a Polars Series of float values, indexed (named) by the quarter
        or month label from the ONS response. Values are sorted chronologically
        and filtered to ``start_date`` or later.

        Routing is handled automatically: SPPI series (e.g. HPTH, HPTD) are
        fetched from the ``/sppi/data`` endpoint; all other series use
        ``/mm23/data``.

        Parameters
        ----------
        series_code:
            ONS series identifier, e.g. ``'HPTH'``. Case-insensitive.
            Use :attr:`CATALOGUE` for the list of recommended codes.
        start_date:
            Earliest date to include in the returned series (ISO format).
            Default is ``'2015-01-01'``.
        frequency:
            Either ``'quarters'`` (default) or ``'months'``. The ONS JSON
            response may contain both; this parameter selects which to use.
        timeout:
            HTTP request timeout in seconds. Default 30.
        cache_path:
            Optional filesystem path to cache the raw JSON response. If the
            file exists it will be read instead of making a network request.
            Set to a temp file path to avoid repeated calls in a session.

        Returns
        -------
        pl.Series
            Named by the ONS series code. Values are the index levels as floats.

        Raises
        ------
        requests.HTTPError
            If the ONS API returns a non-2xx status code.
        ValueError
            If no data matching ``frequency`` is found in the response.
        """
        code = series_code.upper()
        url = _ons_url_for(code)

        # Load from cache if available
        if cache_path and Path(cache_path).exists():
            with open(cache_path) as f:
                data = json.load(f)
        else:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(data, f)

        return cls._parse_ons_response(data, code, frequency, start_date)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        date_col: str,
        value_col: str,
        start_date: Optional[str] = None,
    ) -> pl.Series:
        """Load an external index from a CSV file.

        Use this for BCIS data or any other subscription-based index that
        cannot be fetched via the ONS API.

        Parameters
        ----------
        filepath:
            Path to the CSV file.
        date_col:
            Name of the column containing date or period labels.
        value_col:
            Name of the column containing index values.
        start_date:
            Optional earliest date to include (ISO format string).

        Returns
        -------
        pl.Series
            Named by ``value_col``. Values sorted chronologically.
        """
        df = pl.read_csv(filepath)

        if date_col not in df.columns:
            raise ValueError(f"date_col '{date_col}' not found in {filepath}. "
                             f"Available columns: {df.columns}")
        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in {filepath}. "
                             f"Available columns: {df.columns}")

        # Attempt date filtering
        if start_date:
            try:
                df = df.filter(pl.col(date_col) >= start_date)
            except Exception:  # noqa: BLE001
                warnings.warn(
                    f"Could not filter CSV by start_date='{start_date}'. "
                    "Returning all rows.",
                    UserWarning,
                    stacklevel=2,
                )

        return df[value_col].cast(pl.Float64).rename(value_col)

    @classmethod
    def from_series(
        cls,
        values,
        label: str = "index",
    ) -> pl.Series:
        """Wrap an existing array or Series as an ExternalIndex-compatible Polars Series.

        Parameters
        ----------
        values:
            Array-like, pandas Series, or Polars Series.
        label:
            Name for the returned series.

        Returns
        -------
        pl.Series
        """
        import pandas as pd

        if isinstance(values, pl.Series):
            return values.rename(label)
        if isinstance(values, pd.Series):
            return pl.from_pandas(values.rename(label))
        return pl.Series(label, np.asarray(values, dtype=float))

    @classmethod
    def _parse_ons_response(
        cls,
        data: dict,
        code: str,
        frequency: str,
        start_date: str,
    ) -> pl.Series:
        """Parse the ONS API JSON response into a Polars Series.

        The ONS JSON structure contains a ``quarters`` or ``months`` list,
        each entry having ``date`` and ``value`` fields.
        """
        key = frequency  # 'quarters' or 'months'
        if key not in data:
            # Try the other frequency
            alt = "months" if frequency == "quarters" else "quarters"
            if alt in data:
                warnings.warn(
                    f"ONS series {code}: requested '{frequency}' but only '{alt}' available. "
                    f"Using '{alt}'.",
                    UserWarning,
                    stacklevel=3,
                )
                key = alt
            else:
                available = [k for k in data if k in ("quarters", "months", "years")]
                raise ValueError(
                    f"ONS series {code} response contains no '{frequency}' data. "
                    f"Available: {available}"
                )

        entries = data[key]
        if not entries:
            raise ValueError(f"ONS series {code}: empty '{key}' array in response.")

        periods = []
        values = []
        for entry in entries:
            date_str = entry.get("date", "")
            val_str = entry.get("value", "")
            try:
                val = float(val_str)
            except (ValueError, TypeError):
                continue  # Skip non-numeric entries (e.g. provisional markers)

            # Basic date filter: compare string prefix
            if start_date and date_str < start_date[:7]:  # compare YYYY-MM prefix
                continue

            periods.append(date_str)
            values.append(val)

        if not values:
            raise ValueError(
                f"ONS series {code}: no numeric values found after filtering to start_date='{start_date}'."
            )

        return pl.Series(code, values, dtype=pl.Float64)

    @staticmethod
    def list_catalogue() -> pl.DataFrame:
        """Return a DataFrame listing all catalogued ONS series codes.

        Returns
        -------
        Polars DataFrame with columns ``name``, ``ons_code``.
        """
        return pl.DataFrame(
            {
                "name": list(_CATALOGUE.keys()),
                "ons_code": list(_CATALOGUE.values()),
            }
        )
