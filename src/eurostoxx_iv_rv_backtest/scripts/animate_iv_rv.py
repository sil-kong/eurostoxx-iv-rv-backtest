# src/eurostoxx_iv_rv_backtest/scripts/animate_iv_rv.py
#
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from eurostoxx_iv_rv_backtest.config import OUTPUTS


def main() -> None:
    """
    Anime la volatilité implicite (IV, via VSTOXX) vs
    la volatilité réalisée à 20 jours (RV 20d, annualisée) sur l'Euro STOXX 50.

    - Courbes propres (IV pleine, RV pointillée)
    - Zones colorées:
        rouge = IV > RV  → régime "short vol"
        bleu  = IV < RV  → régime "long vol"
    """

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

    # Décimation: accélère l'animation sans perdre la structure globale
    step = 3  # 1 = chaque jour, 3/5/10 = plus rapide
    x = x[::step]
    iv_pct = iv_pct[::step]
    rv_20_pct = rv_20_pct[::step]

    # ------- Figure & style global -------

    fig, ax = plt.subplots(figsize=(12, 6))

    # Titre principal + sous-titre
    fig.suptitle(
        "Euro STOXX 50 – Implied vs Realized Volatility",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(
        "VSTOXX (IV) vs 20-day realized volatility – regimes long / short vol",
        fontsize=11,
        pad=8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")

    # Grille légère
    ax.grid(True, which="major", alpha=0.25)

    # Axe des dates propre
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # un tick tous les 2 ans
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Limites en x/y
    left = x.min()
    right = x.max()
    ax.set_xlim(left, right)  # type: ignore[arg-type]

    ymax = float(max(iv_pct.max(), rv_20_pct.max()) * 1.1)
    ax.set_ylim(0.0, ymax)

    # ------- Courbes IV / RV -------

    # IV: ligne pleine, couleur "chaude"
    (line_iv,) = ax.plot(
        [],
        [],
        label="Implied vol (VSTOXX, %)",
        linewidth=1.6,
    )

    # RV: ligne pointillée, couleur "froide"
    (line_rv,) = ax.plot(
        [],
        [],
        label="Realized vol 20d (annualized, %)",
        linewidth=1.6,
        linestyle="--",
    )

    # Légende
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Texte explicatif sur les couleurs de régime
    ax.text(
        0.01,
        0.02,
        "Rouge : IV > RV → short vol   |   Bleu : IV < RV → long vol",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.8,
    )

    def _clear_fills() -> None:
        """Supprime tous les fill_between (PolyCollections) existants."""
        # .collections est un ArtistList: pas de .clear() sur certaines versions → remove à la main
        for coll in list(ax.collections):
            coll.remove()

    # ------- Fonctions d'animation -------

    def init():
        """État initial: courbes vides, pas de remplissage."""
        line_iv.set_data([], [])
        line_rv.set_data([], [])
        _clear_fills()
        return line_iv, line_rv

    def update(i: int):
        """
        Mise à jour pour la frame i.

        On dessine:
        - les zones de régime (fill_between rouge / bleu)
        - les courbes IV / RV au-dessus, pour garder une bonne lisibilité
        """
        x_i = x[: i + 1]
        iv_i = iv_pct[: i + 1]
        rv_i = rv_20_pct[: i + 1]

        # Nettoie les anciens remplissages
        _clear_fills()

        # Zones de régime:
        #   rouge = IV > RV → short vol
        ax.fill_between(
            x_i,
            iv_i,
            rv_i,
            where=(iv_i > rv_i),
            alpha=0.25,
            color="red",
            interpolate=True,
        )
        #   bleu = IV < RV → long vol
        ax.fill_between(
            x_i,
            iv_i,
            rv_i,
            where=(iv_i < rv_i),
            alpha=0.25,
            color="blue",
            interpolate=True,
        )

        # Courbes repassées par-dessus les zones
        line_iv.set_data(x_i, iv_i)
        line_rv.set_data(x_i, rv_i)

        return line_iv, line_rv

    # ------- Animation -------

    anim = FuncAnimation(
        fig,
        update,
        frames=range(len(x)),
        init_func=init,
        interval=30,  # ms entre frames
        blit=False,  # blit + fill_between = prise de tête → off
        repeat=False,
    )
    _ = anim  # garde une ref pour éviter le GC / calmer les linters

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # laisse un peu de place au suptitle
    plt.show()


if __name__ == "__main__":
    main()
