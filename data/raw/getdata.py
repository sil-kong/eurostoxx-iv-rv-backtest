from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# --- Paths / constants ---

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

TICKER_SX5E = "^STOXX50E"
V2TX_URL = "https://www.stoxx.com/document/Indices/Current/HistoricalData/h_v2tx.txt"

# --- Period: approx 20 years back ---

end_date = datetime.today()
start_date = end_date - timedelta(days=20 * 365)

# =========================
# 1) EURO STOXX 50 via yfinance
# =========================

print(">>> Downloading SX5E from Yahoo Finance...")

df_sx5e = yf.download(
    TICKER_SX5E,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=False,
    progress=False,
)

if df_sx5e is None or len(df_sx5e) == 0:
    raise RuntimeError(
        f"Aucune donnée renvoyée par yfinance pour le ticker {TICKER_SX5E}."
    )

print("Colonnes brutes SX5E :", list(df_sx5e.columns))

# Flatten colonnes si MultiIndex (cas yfinance)
if isinstance(df_sx5e.columns, pd.MultiIndex):
    df_sx5e.columns = df_sx5e.columns.get_level_values(0)

print("Colonnes après flatten SX5E :", list(df_sx5e.columns))

# On reset l'index -> la colonne 'Date' devient une colonne normale
df_sx5e_raw = df_sx5e.reset_index()

# Sauvegarde RAW (tel que venant de yfinance)
raw_sx5e_path = DATA_RAW / "SXE50_yf_raw.csv"
df_sx5e_raw.to_csv(raw_sx5e_path, index=False)
print(f"[RAW] SX5E sauvegardé dans : {raw_sx5e_path.resolve()}")

# Vérification des colonnes attendues avant renommage
expected_cols = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
missing = expected_cols.difference(set(df_sx5e_raw.columns))
if missing:
    raise RuntimeError(
        f"Colonnes manquantes après reset_index pour SX5E : {missing}\n"
        f"Colonnes réelles : {list(df_sx5e_raw.columns)}"
    )

# DataFrame de travail SX5E
df_sx5e_work = df_sx5e_raw.rename(
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

# date -> datetime (normalement déjà Timestamp, mais on verrouille)
df_sx5e_work["date"] = pd.to_datetime(df_sx5e_work["date"])

if "close" not in df_sx5e_work.columns:
    raise RuntimeError(
        f"La colonne 'close' n'existe pas après renommage. Colonnes = {list(df_sx5e_work.columns)}"
    )

df_sx5e_work = (
    df_sx5e_work.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
)

print("=== SX5E (sous-jacent, fichier de travail) ===")
print(df_sx5e_work.head(5))
print(
    f"\nSX5E lignes: {len(df_sx5e_work)} | "
    f"De {df_sx5e_work['date'].min().date()} à {df_sx5e_work['date'].max().date()}"
)

work_sx5e_path = DATA_RAW / "SXE50_daily_20y.csv"
df_sx5e_work.to_csv(work_sx5e_path, index=False)
print(f"\n Fichier de travail SX5E exporté dans : {work_sx5e_path.resolve()}")


# =========================
# 2) V2TX / VSTOXX (IV du SX5E) via STOXX
# =========================

print("\n>>> Downloading V2TX (VSTOXX) raw txt from STOXX...")

raw_v2tx_txt_path = DATA_RAW / "h_v2tx.txt"
resp = requests.get(V2TX_URL, timeout=20)
resp.raise_for_status()
raw_v2tx_txt_path.write_text(resp.text, encoding="utf-8")
print(f"[RAW] V2TX txt sauvegardé dans : {raw_v2tx_txt_path.resolve()}")

# Lecture du txt brut
df_v2tx_raw = pd.read_csv(
    raw_v2tx_txt_path,
    sep=";",
)

expected_v2tx_cols = {"Date", "Symbol", "Indexvalue"}
missing_v2 = expected_v2tx_cols.difference(set(df_v2tx_raw.columns))
if missing_v2:
    raise RuntimeError(
        f"Colonnes manquantes dans le fichier V2TX : {missing_v2}\n"
        f"Colonnes réelles : {list(df_v2tx_raw.columns)}"
    )

# DataFrame de travail V2TX
df_v2tx_work = df_v2tx_raw.copy()
df_v2tx_work["Date"] = pd.to_datetime(df_v2tx_work["Date"], dayfirst=True)

df_v2tx_work = df_v2tx_work.rename(
    columns={
        "Date": "date",
        "Indexvalue": "vstoxx_close",
    }
)

df_v2tx_work = (
    df_v2tx_work[["date", "vstoxx_close"]]
    .dropna()
    .sort_values("date")
    .reset_index(drop=True)
)

# IV en décimal (0.20 pour 20 %)
df_v2tx_work["iv"] = df_v2tx_work["vstoxx_close"] / 100.0

print("\n=== V2TX (IV, fichier de travail avant filtre) ===")
print(df_v2tx_work.head(5))

work_v2tx_path = DATA_RAW / "V2TX_full_daily.csv"
df_v2tx_work.to_csv(work_v2tx_path, index=False)
print(f"\n Fichier de travail V2TX (full) exporté dans : {work_v2tx_path.resolve()}")

# =========================
# 3) FICHIER DE TRAVAIL FUSIONNÉ SX5E + IV
# =========================

# On restreint V2TX à la période SX5E pour être cohérent
start_p = df_sx5e_work["date"].min()
end_p = df_sx5e_work["date"].max()
mask = (df_v2tx_work["date"] >= start_p) & (df_v2tx_work["date"] <= end_p)
df_v2tx_20y = df_v2tx_work.loc[mask].reset_index(drop=True)

print(
    f"\nV2TX filtré sur période SX5E : {len(df_v2tx_20y)} lignes | "
    f"De {df_v2tx_20y['date'].min().date()} à {df_v2tx_20y['date'].max().date()}"
)

# 🔹 Merge sur 'date' (les deux colonnes sont en datetime -> format interne, pas un problème)
df_merged = pd.merge(
    df_sx5e_work,
    df_v2tx_20y[["date", "vstoxx_close", "iv"]],
    on="date",
    how="left",  # left: on garde toutes les dates SX5E, IV peut être NaN au début
)

print("\n=== Fichier de travail MERGÉ SX5E + IV ===")
print(df_merged.head(5))

merged_path = DATA_RAW / "SXE50_with_IV_daily_20y.csv"
df_merged.to_csv(merged_path, index=False)
print(f"\n Fichier de travail fusionné exporté dans : {merged_path.resolve()}")
