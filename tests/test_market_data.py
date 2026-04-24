from datetime import date

import pandas as pd
import pytest

from eurostoxx_iv_rv_backtest.data.market_data import (
    clean_sx5e_data,
    clean_vstoxx_data,
    merge_spot_and_iv,
    resolve_date_range,
)


def test_clean_sx5e_data_renames_sorts_and_validates_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02"],
            "Open": [101.0, 100.0],
            "High": [102.0, 101.0],
            "Low": [100.0, 99.0],
            "Close": [101.5, 100.5],
            "Adj Close": [101.5, 100.5],
            "Volume": [1000, 900],
        }
    )

    clean = clean_sx5e_data(raw)

    assert list(clean.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert clean["date"].is_monotonic_increasing
    assert clean.loc[0, "close"] == 100.5


def test_clean_vstoxx_data_adds_decimal_iv() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["03.01.2024", "02.01.2024"],
            "Symbol": ["V2TX", "V2TX"],
            "Indexvalue": [18.0, 20.0],
        }
    )

    clean = clean_vstoxx_data(raw)

    assert clean["date"].is_monotonic_increasing
    assert clean.loc[0, "iv"] == 0.20
    assert clean.loc[1, "vstoxx_close"] == 18.0


def test_merge_spot_and_iv_left_joins_and_checks_missing_ratio() -> None:
    sx5e = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    vstoxx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "vstoxx_close": [20.0, 22.0],
            "iv": [0.20, 0.22],
        }
    )

    merged = merge_spot_and_iv(sx5e, vstoxx, max_missing_iv_ratio=0.50)

    assert len(merged) == 3
    assert merged.loc[0, "iv"] == 0.20
    assert pd.isna(merged.loc[1, "iv"])
    assert merged.loc[2, "iv"] == 0.22


def test_merge_spot_and_iv_rejects_duplicate_dates() -> None:
    sx5e = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "close": [100.0, 101.0],
        }
    )
    vstoxx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "vstoxx_close": [20.0],
            "iv": [0.20],
        }
    )

    with pytest.raises(RuntimeError, match="dupliquees"):
        merge_spot_and_iv(sx5e, vstoxx)


def test_resolve_date_range_preserves_20y_default_when_only_end_is_given() -> None:
    start, end = resolve_date_range(None, date(2024, 1, 10))

    assert end == date(2024, 1, 10)
    assert (end - start).days == 20 * 365
