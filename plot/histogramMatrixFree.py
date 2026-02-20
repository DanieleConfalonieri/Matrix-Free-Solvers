import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Iterable
from matplotlib.ticker import ScalarFormatter


def _pick_unit_and_scale(values_seconds: np.ndarray):
    v = values_seconds[np.isfinite(values_seconds)]
    v = v[v > 0]
    if v.size == 0:
        return 1.0, "s"
    m = np.median(v)
    if m < 1e-6:
        return 1e9, "ns"
    if m < 1e-3:
        return 1e6, "µs"
    if m < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def _format_time(val: float) -> str:
    if val >= 1000:
        return f"{val:.0f}"
    if val >= 100:
        return f"{val:.1f}"
    if val >= 10:
        return f"{val:.2f}"
    return f"{val:.3f}"


def plot_matrixfree_56cores_by_refinements(
    csv_path: str,
    time_kind: str = "Wall",
    agg: str = "mean",
    cores: int = 56,
    refinements: Iterable[int] = (3, 4),
    title: Optional[str] = None,
):
    df = pd.read_csv(csv_path)

    setup_col = f"SetupSystem{time_kind}"
    solve_col = f"SolvingLinearSystem{time_kind}"

    required = {"Cores", "Solver", "Degree", "Refinement", setup_col, solve_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    # --- Filtra SOLO MatrixFree (case-insensitive) e SOLO cores richiesti
    df = df.copy()
    sraw = df["Solver"].astype(str).str.strip().str.lower()
    df = df[(sraw == "matrixfree") & (df["Cores"] == cores)].copy()
    if df.empty:
        available_cores = sorted(pd.read_csv(csv_path)["Cores"].unique())
        raise ValueError(
            f"Nessuna riga con Solver=MatrixFree e Cores={cores}. "
            f"Cores disponibili nel file: {available_cores}"
        )

    # --- Stile
    COLOR_FREE = "#0b3d91"   # blu profondo
    bar_width = 0.55

    # --- Scala/unità coerente tra i plot (stimata su tutte le refinements richieste)
    df_all_refs = df[df["Refinement"].isin(list(refinements))].copy()
    if df_all_refs.empty:
        available_refs = sorted(df["Refinement"].unique())
        raise ValueError(
            f"Nessuna riga per Refinement in {list(refinements)} con Cores={cores} e Solver=MatrixFree. "
            f"Refinement disponibili: {available_refs}"
        )

    if agg == "sum":
        tmp = df_all_refs.groupby(["Degree", "Refinement"], as_index=False)[[setup_col, solve_col]].sum()
    else:
        tmp = df_all_refs.groupby(["Degree", "Refinement"], as_index=False)[[setup_col, solve_col]].mean()

    values_seconds = np.concatenate([tmp[setup_col].to_numpy(), tmp[solve_col].to_numpy()])
    scale, unit = _pick_unit_and_scale(values_seconds)

    # --- Un plot per ciascun refinement richiesto
    for ref in refinements:
        dfr = df[df["Refinement"] == ref].copy()
        if dfr.empty:
            continue

        # Aggregazione per Degree (Ref fissato)
        if agg == "sum":
            dfg = dfr.groupby(["Degree", "Refinement"], as_index=False)[[setup_col, solve_col]].sum()
        elif agg == "mean":
            dfg = dfr.groupby(["Degree", "Refinement"], as_index=False)[[setup_col, solve_col]].mean()
        else:
            raise ValueError("agg deve essere 'mean' oppure 'sum'")

        # Applica scala comune
        dfg[setup_col] *= scale
        dfg[solve_col] *= scale

        # Ordina per Degree
        dfg = dfg.sort_values(["Degree"], kind="mergesort").reset_index(drop=True)

        x = np.arange(len(dfg))
        labels = [f"Deg {d}" for d in dfg["Degree"]]

        # ==========================================================
        # Plot 1: tempi assoluti (Setup bottom + Solve top), log scale
        # ==========================================================
        fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(dfg)), 6))

        ax.bar(x, dfg[setup_col], width=bar_width, color=COLOR_FREE, alpha=0.35, label="MatrixFree - setup")
        ax.bar(x, dfg[solve_col], width=bar_width, bottom=dfg[setup_col], color=COLOR_FREE, alpha=1.0,
               label="MatrixFree - solve")

        totals = (dfg[setup_col] + dfg[solve_col]).to_numpy()
        for xi, total in zip(x, totals):
            if total > 0:
                ax.text(xi, total * 1.12, _format_time(float(total)), ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel("Degree")

        ax.set_yscale("log")
        ax.set_ylabel(f"Time ({unit}, log scale)")

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)

        if title is None:
            ax.set_title(f"Setup + Solve — MatrixFree | Cores = {cores} | Refinement = {ref}")
        else:
            ax.set_title(f"{title} | MatrixFree | Cores = {cores} | Refinement = {ref}")

        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ============================================
        # Plot 2: composizione (frazioni) Setup vs Solve
        # ============================================
        setup_vals = dfg[setup_col].to_numpy(dtype=float)
        solve_vals = dfg[solve_col].to_numpy(dtype=float)
        total_vals = setup_vals + solve_vals

        safe_total = np.where(total_vals > 0, total_vals, np.nan)
        setup_frac = setup_vals / safe_total
        solve_frac = solve_vals / safe_total

        fig2, ax2 = plt.subplots(figsize=(max(10, 0.9 * len(dfg)), 4.8))
        ax2.bar(x, setup_frac, width=bar_width, color=COLOR_FREE, alpha=0.35, label="Setup fraction")
        ax2.bar(x, solve_frac, width=bar_width, bottom=setup_frac, color=COLOR_FREE, alpha=1.0, label="Solve fraction")

        # percentuale solve sopra (facoltativo ma utile)
        for xi, sf in zip(x, solve_frac):
            if np.isfinite(sf):
                ax2.text(xi, 1.02, f"{sf*100:.0f}%", ha="center", va="bottom", fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=0, ha="center")
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("Fraction of total time")
        ax2.set_xlabel("Degree")
        ax2.set_title(f"Setup vs Solve composition — MatrixFree | Cores = {cores} | Refinement = {ref}")

        ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax2.legend(ncols=2)

        plt.tight_layout()
        plt.show()

        # ============================================
        # Plot 3: rapporto Solve / Setup (solo rapporto)
        # ============================================
        ratio = np.divide(
            solve_vals,
            setup_vals,
            out=np.full_like(solve_vals, np.nan),
            where=setup_vals > 0
        )

        fig3, ax3 = plt.subplots(figsize=(max(10, 0.9 * len(dfg)), 4.5))
        ax3.plot(x, ratio, marker="o", linewidth=2, color=COLOR_FREE)

        # riferimento ratio = 1
        ax3.axhline(1.0, linestyle="--", linewidth=1)

        # etichette sul rapporto
        for xi, r in zip(x, ratio):
            if np.isfinite(r):
                ax3.text(xi, r * 1.05, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=0, ha="center")
        ax3.set_xlabel("Degree")
        ax3.set_ylabel("Solve / Setup ratio")
        ax3.set_title(f"Solve-to-Setup ratio — MatrixFree | Cores = {cores} | Refinement = {ref}")

        ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    csv_file = "MatrixFree_p12.csv"

    plot_matrixfree_56cores_by_refinements(
        csv_path=csv_file,
        time_kind="Wall",
        agg="mean",
        cores=56,
        refinements=(3, 4),
        title=None
    )