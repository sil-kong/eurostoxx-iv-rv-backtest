# src/eurostoxx_iv_rv_backtest/scripts/animate_iv_rv.py

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    csv_path = OUTPUTS / "SXE50_with_IV_RV_daily_20y.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable.\n"
            "Lance d'abord build_rv.py pour générer les données."
        )

    print(f">>> Lecture de {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # x = dates, y1 = IV en %, y2 = RV 20j en %
    x = df["date"]
    iv_pct = df["iv"] * 100.0
    rv_20_pct = df["rv_20d_pct"]

    # Petite décimation pour que l’anim ne rame pas trop
    step = 3  # 1 = frame par jour, 3/5/10 = plus rapide
    x = x[::step]
    iv_pct = iv_pct[::step]
    rv_20_pct = rv_20_pct[::step]

    # Figure
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Euro STOXX 50 – Implied vs 20d Realized Vol (animation)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")

    # Limites en x/y
    left = x.min()
    right = x.max()
    ax.set_xlim(left, right)  # type: ignore[arg-type]

    ymax = float(max(iv_pct.max(), rv_20_pct.max()) * 1.1)
    ax.set_ylim(0.0, ymax)

    # Deux courbes vides au départ
    (line_iv,) = ax.plot([], [], label="IV (VSTOXX, %)")
    (line_rv,) = ax.plot([], [], label="RV 20d (%, annualisée)")
    ax.legend()

    def _clear_fills() -> None:
        """Supprime tous les fill_between (PolyCollections) existants."""
        # .collections est un ArtistList, pas toujours avec .clear() -> on remove à la main
        for coll in list(ax.collections):
            coll.remove()

    # Fonction d’init
    def init():
        line_iv.set_data([], [])
        line_rv.set_data([], [])
        _clear_fills()
        return line_iv, line_rv

    # Fonction appelée à chaque frame
    def update(i: int):
        # i = index de frame
        x_i = x[: i + 1]
        iv_i = iv_pct[: i + 1]
        rv_i = rv_20_pct[: i + 1]

        # Mise à jour des lignes
        line_iv.set_data(x_i, iv_i)
        line_rv.set_data(x_i, rv_i)

        # On enlève les anciens fill_between pour ne pas les empiler
        _clear_fills()

        # Zones :
        #   rouge = IV > RV  → short vol
        #   bleu  = IV < RV  → long vol
        ax.fill_between(
            x_i,
            iv_i,
            rv_i,
            where=(iv_i > rv_i),
            alpha=0.25,
            color="red",
            interpolate=True,
        )
        ax.fill_between(
            x_i,
            iv_i,
            rv_i,
            where=(iv_i < rv_i),
            alpha=0.25,
            color="blue",
            interpolate=True,
        )

        return line_iv, line_rv

    anim = FuncAnimation(
        fig,
        update,
        frames=range(len(x)),
        init_func=init,
        interval=30,  # ms entre frames
        blit=False,  # blit + fill_between = chiant, donc off
        repeat=False,
    )
    _ = anim  # garder une ref pour éviter le GC / calmer le linter

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
