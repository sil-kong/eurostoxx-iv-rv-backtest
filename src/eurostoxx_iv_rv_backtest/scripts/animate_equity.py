# src/eurostoxx_iv_rv_backtest/scripts/animate_equity.py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eurostoxx_iv_rv_backtest.config import OUTPUTS


def plot_equity(show_regimes: bool = True) -> None:
    """
    Trace une equity curve propre pour la stratégie variance swap IV vs RV.

    Lit : OUTPUTS / 'SXE50_iv_rv_varswap_backtest.csv'
    - Courbe : equity_varswap
    - Fond (optionnel) : blocs rouges / bleus selon signal_vol
      rouge = short vol, bleu = long vol
    """

    csv_path = OUTPUTS / "SXE50_iv_rv_varswap_backtest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable.\nLance d'abord run_backtest_iv_rv.py."
        )

    print(f">>> Lecture de {csv_path}")
    df = (
        pd.read_csv(csv_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    required_cols = ["date", "equity_varswap"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"Colonne '{c}' manquante. Colonnes dispo : {list(df.columns)}"
            )

    if "signal_vol" in df.columns:
        df["signal_vol"] = df["signal_vol"].fillna(0).astype(int)
    else:
        df["signal_vol"] = 0

    # On coupe la phase où l’equity est strictement à 0 (phase de chauffe)
    first_move_idx = (df["equity_varswap"] != 0).idxmax()
    if first_move_idx > 0:
        df = df.iloc[first_move_idx:].reset_index(drop=True)

    x = df["date"]
    equity = df["equity_varswap"]
    signal = df["signal_vol"]

    # ---------- Figure & style ---------- #

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    fig.suptitle(
        "Euro STOXX 50 – IV vs RV variance swap strategy",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(
        "Cumulative PnL (notional = 1) – long / short volatility regimes",
        fontsize=11,
        pad=8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")

    # Axe des dates: ticks tous les 2 ans
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Limites X
    ax.set_xlim(x.iloc[0], x.iloc[-1])
    # Limites Y avec petite marge

    # On passe par un array numpy pour calmer le moteur de typage
    eq_arr = equity.to_numpy(dtype="float64")

    eq_min_val = float(np.nanmin(eq_arr))
    eq_max_val = float(np.nanmax(eq_arr))

    y_min = min(eq_min_val, 0.0)
    y_max = max(eq_max_val, 0.0)

    span = (y_max - y_min) or 1.0
    margin = 0.07 * span
    ax.set_ylim(y_min - margin, y_max + margin)

    # Ligne zéro
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)

    # ---------- Blocs de régimes ---------- #

    if show_regimes:
        min_len = 10  # jours min pour afficher un bloc (évite l’effet code-barres)

        prev_regime = signal.iloc[0]
        start_idx = 0

        for i in range(1, len(df)):
            regime = signal.iloc[i]
            if regime != prev_regime:
                length = i - start_idx
                if prev_regime != 0 and length >= min_len:
                    start_date = x.iloc[start_idx]
                    end_date = x.iloc[i - 1]
                    color = "red" if prev_regime < 0 else "blue"
                    ax.axvspan(
                        start_date,
                        end_date,
                        color=color,
                        alpha=0.08,
                        linewidth=0,
                    )
                start_idx = i
                prev_regime = regime

        # Dernier segment
        length = len(df) - start_idx
        if prev_regime != 0 and length >= min_len:
            start_date = x.iloc[start_idx]
            end_date = x.iloc[len(df) - 1]
            color = "red" if prev_regime < 0 else "blue"
            ax.axvspan(
                start_date,
                end_date,
                color=color,
                alpha=0.08,
                linewidth=0,
            )

        ax.text(
            0.01,
            0.02,
            "Rouge : short vol   |   Bleu : long vol",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.75,
        )

    # ---------- Courbe d’equity ---------- #

    ax.plot(
        x,
        equity,
        linewidth=2.0,
        color="#0055aa",
        label="Equity IV–RV varswap",
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    plot_equity()
