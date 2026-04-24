from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests
import yfinance as yf


TICKER_SX5E = "^STOXX50E"
V2TX_URL = "https://www.stoxx.com/document/Indices/Current/HistoricalData/h_v2tx.txt"
DEFAULT_LOOKBACK_DAYS = 20 * 365


@dataclass(frozen=True)
class DatasetSummary:
    """Small validation summary printed by the ingestion pipeline."""

    label: str
    rows: int
    start_date: date
    end_date: date
    missing_iv_ratio: float | None = None


def default_date_range(today: date | None = None) -> tuple[date, date]:
    """Return the current default period: roughly 20 calendar years ending today."""
    end = today or datetime.today().date()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start, end


def parse_cli_date(value: str | None) -> date | None:
    """Parse an optional YYYY-MM-DD CLI date."""
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_date_range(start: date | None, end: date | None) -> tuple[date, date]:
    """
    Resolve optional CLI dates into an explicit start/end period.

    If neither date is supplied, this preserves the historical behavior of using
    approximately 20 calendar years ending at the current date. If only ``end``
    is supplied, ``start`` is inferred from the same 20-year lookback.
    """
    default_start, default_end = default_date_range()
    resolved_end = end or default_end
    resolved_start = start or (resolved_end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    if resolved_start >= resolved_end:
        raise ValueError("La date --start doit être strictement antérieure à --end.")
    return resolved_start, resolved_end


def fetch_sx5e_data(
    start: date,
    end: date,
    ticker: str = TICKER_SX5E,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download raw Euro STOXX 50 spot data from Yahoo Finance."""
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or len(df) == 0:
        raise RuntimeError(f"Aucune donnée renvoyée par yfinance pour le ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.reset_index()


def fetch_vstoxx_data(
    output_dir: Path,
    url: str = V2TX_URL,
    force_download: bool = False,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Fetch or reuse the raw VSTOXX historical text file and return it as a DataFrame.

    ``force_download=False`` reuses ``h_v2tx.txt`` when present, which makes
    repeated local pipeline runs less dependent on the network.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_txt_path = output_dir / "h_v2tx.txt"

    if force_download or not raw_txt_path.exists():
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        raw_txt_path.write_text(response.text, encoding="utf-8")
        print(f"[RAW] V2TX txt saved to: {raw_txt_path.resolve()}")
    else:
        print(f"[RAW] Reusing cached V2TX txt: {raw_txt_path.resolve()}")

    return pd.read_csv(raw_txt_path, sep=";")


def clean_sx5e_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Yahoo Finance SX5E data into the project column convention."""
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    _require_columns(df_raw, expected_cols, "SX5E raw")

    df = df_raw.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    _validate_time_series(df, "SX5E clean")
    return df


def clean_vstoxx_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw STOXX V2TX data and add decimal implied volatility."""
    expected_cols = {"Date", "Symbol", "Indexvalue"}
    _require_columns(df_raw, expected_cols, "VSTOXX raw")

    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.rename(columns={"Date": "date", "Indexvalue": "vstoxx_close"})
    df["iv"] = df["vstoxx_close"] / 100.0
    df = df.sort_values("date").reset_index(drop=True)
    _validate_time_series(df, "VSTOXX clean")
    return df


def merge_spot_and_iv(
    sx5e: pd.DataFrame,
    vstoxx: pd.DataFrame,
    max_missing_iv_ratio: float = 0.10,
) -> pd.DataFrame:
    """Merge cleaned spot and VSTOXX data on date with left join on spot dates."""
    _require_columns(sx5e, {"date", "close"}, "SX5E clean")
    _require_columns(vstoxx, {"date", "vstoxx_close", "iv"}, "VSTOXX clean")
    _validate_time_series(sx5e, "SX5E clean")
    _validate_time_series(vstoxx, "VSTOXX clean")

    start = sx5e["date"].min()
    end = sx5e["date"].max()
    vstoxx_period = vstoxx.loc[
        (vstoxx["date"] >= start) & (vstoxx["date"] <= end)
    ].reset_index(drop=True)

    merged = pd.merge(
        sx5e,
        vstoxx_period[["date", "vstoxx_close", "iv"]],
        on="date",
        how="left",
    )
    _validate_time_series(merged, "SX5E + VSTOXX merged")

    missing_iv_ratio = float(merged["iv"].isna().mean())
    if missing_iv_ratio >= 1.0:
        raise RuntimeError("La fusion SX5E/VSTOXX ne contient aucune IV disponible.")
    if missing_iv_ratio > max_missing_iv_ratio:
        raise RuntimeError(
            "Ratio d'IV manquante trop élevé après fusion: "
            f"{missing_iv_ratio:.2%} > {max_missing_iv_ratio:.2%}."
        )

    return merged


def build_raw_dataset(
    start: date | None = None,
    end: date | None = None,
    output_dir: Path | str = Path("data/raw"),
    force_download: bool = False,
    ticker: str = TICKER_SX5E,
    vstoxx_url: str = V2TX_URL,
) -> pd.DataFrame:
    """
    Build and export the raw working dataset used by the research pipeline.

    Output semantics are preserved from the original script:
        - SXE50_yf_raw.csv
        - SXE50_daily_20y.csv
        - h_v2tx.txt
        - V2TX_full_daily.csv
        - SXE50_with_IV_daily_20y.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_start, resolved_end = resolve_date_range(start, end)
    print(
        ">>> Building raw dataset for "
        f"{resolved_start.isoformat()} to {resolved_end.isoformat()}"
    )

    print(">>> Downloading SX5E from Yahoo Finance...")
    sx5e_raw = fetch_sx5e_data(resolved_start, resolved_end, ticker=ticker)
    sx5e_raw_path = output_path / "SXE50_yf_raw.csv"
    sx5e_raw.to_csv(sx5e_raw_path, index=False)
    print(f"[RAW] SX5E saved to: {sx5e_raw_path.resolve()}")

    sx5e_clean = clean_sx5e_data(sx5e_raw)
    sx5e_clean_path = output_path / "SXE50_daily_20y.csv"
    sx5e_clean.to_csv(sx5e_clean_path, index=False)
    _print_summary(_summarize(sx5e_clean, "SX5E clean"))
    print(f"[WORK] SX5E saved to: {sx5e_clean_path.resolve()}")

    print(">>> Loading V2TX (VSTOXX) data...")
    vstoxx_raw = fetch_vstoxx_data(
        output_path,
        url=vstoxx_url,
        force_download=force_download,
    )
    vstoxx_clean = clean_vstoxx_data(vstoxx_raw)
    vstoxx_clean_path = output_path / "V2TX_full_daily.csv"
    vstoxx_clean.to_csv(vstoxx_clean_path, index=False)
    _print_summary(_summarize(vstoxx_clean, "VSTOXX clean"))
    print(f"[WORK] VSTOXX full saved to: {vstoxx_clean_path.resolve()}")

    merged = merge_spot_and_iv(sx5e_clean, vstoxx_clean)
    merged_path = output_path / "SXE50_with_IV_daily_20y.csv"
    merged.to_csv(merged_path, index=False)
    _print_summary(
        _summarize(
            merged,
            "SX5E + VSTOXX merged",
            missing_iv_ratio=float(merged["iv"].isna().mean()),
        )
    )
    print(f"[WORK] Merged dataset saved to: {merged_path.resolve()}")

    return merged


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for building the raw market dataset."""
    parser = argparse.ArgumentParser(description="Build SX5E/VSTOXX raw dataset.")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", help="End date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where raw and working CSV files are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download VSTOXX text data even if a cached file exists.",
    )
    parser.add_argument("--ticker", default=TICKER_SX5E, help="Yahoo Finance ticker.")
    args = parser.parse_args(argv)

    build_raw_dataset(
        start=parse_cli_date(args.start),
        end=parse_cli_date(args.end),
        output_dir=Path(args.output_dir),
        force_download=args.force_download,
        ticker=args.ticker,
    )


def _require_columns(df: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = expected.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"Colonnes manquantes pour {label}: {sorted(missing)}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )


def _validate_time_series(df: pd.DataFrame, label: str) -> None:
    _require_columns(df, {"date"}, label)
    if df.empty:
        raise RuntimeError(f"{label} ne contient aucune ligne.")
    if df["date"].duplicated().any():
        duplicate_dates = df.loc[df["date"].duplicated(), "date"].dt.date.unique()
        raise RuntimeError(f"{label} contient des dates dupliquees: {duplicate_dates[:5]}")
    if not df["date"].is_monotonic_increasing:
        raise RuntimeError(f"{label} n'est pas trie par date croissante.")


def _summarize(
    df: pd.DataFrame,
    label: str,
    missing_iv_ratio: float | None = None,
) -> DatasetSummary:
    return DatasetSummary(
        label=label,
        rows=len(df),
        start_date=df["date"].min().date(),
        end_date=df["date"].max().date(),
        missing_iv_ratio=missing_iv_ratio,
    )


def _print_summary(summary: DatasetSummary) -> None:
    line = (
        f"{summary.label}: {summary.rows} rows | "
        f"{summary.start_date.isoformat()} to {summary.end_date.isoformat()}"
    )
    if summary.missing_iv_ratio is not None:
        line += f" | missing IV: {summary.missing_iv_ratio:.2%}"
    print(line)


if __name__ == "__main__":
    main()
