import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch


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


def plot_histogram_selected_cores(
    csv_path: str,
    time_kind: str = "Wall",
    agg: str = "mean",
    selected_cores=(56, 28, 10),
    title: Optional[str] = None,
):
    df = pd.read_csv(csv_path)

    setup_col = f"SetupSystem{time_kind}"
    solve_col = f"SolvingLinearSystem{time_kind}"

    required = {"Cores", "Solver", "Degree", "Refinement", setup_col, solve_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Normalize solver
    df = df.copy()
    sraw = df["Solver"].astype(str).str.strip().str.lower()
    df["Solver_norm"] = np.where(
        sraw == "matrixfree",
        "matrix_free",
        np.where(sraw == "matrixbased", "matrix_based", "other")
    )
    df = df[df["Solver_norm"].isin(["matrix_free", "matrix_based"])].copy()
    if df.empty:
        raise ValueError("No rows found with Solver=MatrixFree/MatrixBased in CSV.")

    # Colors
    COLOR_FREE = "#0b3d91"   # deep blue
    COLOR_BASED = "#7aa6ff"  # light blue

    available = set(df["Cores"].unique())
    cores_to_plot = [c for c in selected_cores if c in available]
    if not cores_to_plot:
        raise ValueError(
            f"None of the requested cores {selected_cores} are present in the CSV. Available: {sorted(available)}"
        )

    for cores in cores_to_plot:
        dfi = df[df["Cores"] == cores].copy()

        # Aggregation
        group_cols = ["Solver_norm", "Degree", "Refinement"]
        if agg == "sum":
            dfg = dfi.groupby(group_cols, as_index=False)[[setup_col, solve_col]].sum()
        elif agg == "mean":
            dfg = dfi.groupby(group_cols, as_index=False)[[setup_col, solve_col]].mean()
        else:
            raise ValueError("agg must be 'mean' or 'sum'")

        # Scale/unit
        values_seconds = np.concatenate([dfg[setup_col].to_numpy(), dfg[solve_col].to_numpy()])
        scale, unit = _pick_unit_and_scale(values_seconds)
        dfg[setup_col] *= scale
        dfg[solve_col] *= scale

        # Categories
        cats = (
            dfg[["Degree", "Refinement"]]
            .drop_duplicates()
            .sort_values(["Degree", "Refinement"], kind="mergesort")
            .reset_index(drop=True)
        )
        x = np.arange(len(cats))
        cat_labels = [f"Deg {d} | Ref {r}" for d, r in zip(cats["Degree"], cats["Refinement"])]

        def get_vals(solver_norm, deg, ref):
            hit = dfg[
                (dfg["Solver_norm"] == solver_norm) &
                (dfg["Degree"] == deg) &
                (dfg["Refinement"] == ref)
            ]
            if hit.empty:
                return None
            row = hit.iloc[0]
            return float(row[setup_col]), float(row[solve_col])

        bar_width = 0.36
        gap = 0.08

        # ==========================================================
        # Plot 1: absolute times
        # ==========================================================
        fig1, ax1 = plt.subplots(figsize=(max(12, 1.0 * len(cats)), 6))
        labeled = set()

        def add_stacked_bar(ax, xpos, setup, solve, label, color):
            l1 = f"{label} - setup"
            ax.bar(xpos, setup, width=bar_width, color=color, alpha=0.35,
                   label=(l1 if l1 not in labeled else None))
            labeled.add(l1)

            l2 = f"{label} - solve"
            ax.bar(xpos, solve, width=bar_width, bottom=setup, color=color, alpha=1.0,
                   label=(l2 if l2 not in labeled else None))
            labeled.add(l2)

            total = setup + solve
            if total > 0:
                ax.text(xpos, total * 1.12, _format_time(total), ha="center", va="bottom", fontsize=9)

        for i, (deg, ref) in enumerate(zip(cats["Degree"], cats["Refinement"])):
            based = get_vals("matrix_based", deg, ref)
            free = get_vals("matrix_free", deg, ref)

            if based is not None and free is not None:
                xl = x[i] - (bar_width / 2.0 + gap / 2.0)
                xr = x[i] + (bar_width / 2.0 + gap / 2.0)
                add_stacked_bar(ax1, xl, based[0], based[1], "MatrixBased", COLOR_BASED)
                add_stacked_bar(ax1, xr, free[0], free[1], "MatrixFree", COLOR_FREE)
            elif free is not None:
                add_stacked_bar(ax1, x[i], free[0], free[1], "MatrixFree", COLOR_FREE)
            elif based is not None:
                add_stacked_bar(ax1, x[i], based[0], based[1], "MatrixBased", COLOR_BASED)

        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_labels, rotation=45, ha="right")
        ax1.set_xlabel("Degree | Refinement")
        ax1.set_yscale("log")
        ax1.set_ylabel(f"Time ({unit}, log scale)")

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax1.yaxis.set_major_formatter(fmt)

        if title is None:
            ax1.set_title(f"Setup + Solve — MatrixBased vs MatrixFree | Cores = {cores}")
        else:
            ax1.set_title(f"{title} | Cores = {cores}")

        ax1.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax1.legend(ncols=2)
        plt.tight_layout()
        plt.show()

        # ==========================================================
        # Plot 2: composition (fractions)
        # ==========================================================
        setup_based, solve_based, setup_free, solve_free = [], [], [], []
        for deg, ref in zip(cats["Degree"], cats["Refinement"]):
            b = get_vals("matrix_based", deg, ref)
            f = get_vals("matrix_free", deg, ref)
            if b is None:
                setup_based.append(np.nan); solve_based.append(np.nan)
            else:
                setup_based.append(b[0]); solve_based.append(b[1])
            if f is None:
                setup_free.append(np.nan); solve_free.append(np.nan)
            else:
                setup_free.append(f[0]); solve_free.append(f[1])

        setup_based = np.array(setup_based, dtype=float)
        solve_based = np.array(solve_based, dtype=float)
        setup_free = np.array(setup_free, dtype=float)
        solve_free = np.array(solve_free, dtype=float)

        fig2, ax2 = plt.subplots(figsize=(max(12, 1.0 * len(cats)), 4.8))

        for i, (deg, ref) in enumerate(zip(cats["Degree"], cats["Refinement"])):
            b_setup = setup_based[i]; b_solve = solve_based[i]
            f_setup = setup_free[i];  f_solve = solve_free[i]

            if np.isfinite(b_setup) and np.isfinite(b_solve):
                total_b = b_setup + b_solve
                sb = b_setup / total_b if total_b > 0 else np.nan
                vb = b_solve / total_b if total_b > 0 else np.nan
                xl = x[i] - (bar_width / 2.0 + gap / 2.0) if (np.isfinite(f_setup) and np.isfinite(f_solve)) else x[i]
                ax2.bar(xl, sb, width=bar_width, color=COLOR_BASED, alpha=0.35)
                ax2.bar(xl, vb, width=bar_width, bottom=sb, color=COLOR_BASED, alpha=1.0)

            if np.isfinite(f_setup) and np.isfinite(f_solve):
                total_f = f_setup + f_solve
                sf = f_setup / total_f if total_f > 0 else np.nan
                vf = f_solve / total_f if total_f > 0 else np.nan
                xr = x[i] + (bar_width / 2.0 + gap / 2.0) if (np.isfinite(b_setup) and np.isfinite(b_solve)) else x[i]
                ax2.bar(xr, sf, width=bar_width, color=COLOR_FREE, alpha=0.35)
                ax2.bar(xr, vf, width=bar_width, bottom=sf, color=COLOR_FREE, alpha=1.0)

        ax2.set_xticks(x)
        ax2.set_xticklabels(cat_labels, rotation=45, ha="right")
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("Fraction of total time")
        ax2.set_xlabel("Degree | Refinement")
        ax2.set_title(f"Setup vs Solve composition (fractions) | Cores = {cores}")

        # FIXED LEGEND (Manual patches)
        legend_handles = [
            Patch(facecolor=COLOR_BASED, edgecolor="none", alpha=0.35, label="MatrixBased - setup frac"),
            Patch(facecolor=COLOR_BASED, edgecolor="none", alpha=1.00, label="MatrixBased - solve frac"),
            Patch(facecolor=COLOR_FREE,  edgecolor="none", alpha=0.35, label="MatrixFree - setup frac"),
            Patch(facecolor=COLOR_FREE,  edgecolor="none", alpha=1.00, label="MatrixFree - solve frac"),
        ]
        ax2.legend(handles=legend_handles, ncols=2, frameon=True)

        ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()
        plt.show()

        # ==========================================================
        # Plot 3: Solve / Setup ratio
        # ==========================================================
        def ratio(solve_arr, setup_arr):
            return np.divide(
                solve_arr, setup_arr,
                out=np.full_like(solve_arr, np.nan),
                where=np.isfinite(setup_arr) & (setup_arr > 0)
            )

        ratio_based = ratio(solve_based, setup_based)
        ratio_free = ratio(solve_free, setup_free)

        fig3, ax3 = plt.subplots(figsize=(max(12, 1.0 * len(cats)), 4.8))
        ax3.plot(x, ratio_based, marker="o", linewidth=2, color=COLOR_BASED, label="MatrixBased: solve/setup")
        ax3.plot(x, ratio_free, marker="o", linewidth=2, color=COLOR_FREE, label="MatrixFree: solve/setup")

        ax3.axhline(1.0, linestyle="--", linewidth=1)

        for xi, r in zip(x, ratio_based):
            if np.isfinite(r):
                ax3.text(xi, r * 1.05, f"{r:.2f}", ha="center", va="bottom", fontsize=9, color=COLOR_BASED)
        for xi, r in zip(x, ratio_free):
            if np.isfinite(r):
                ax3.text(xi, r * 1.05, f"{r:.2f}", ha="center", va="bottom", fontsize=9, color=COLOR_FREE)

        ax3.set_xticks(x)
        ax3.set_xticklabels(cat_labels, rotation=45, ha="right")
        ax3.set_xlabel("Degree | Refinement")
        ax3.set_ylabel("Solve / Setup ratio")
        ax3.set_title(f"Solve-to-Setup ratio | Cores = {cores}")
        ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    csv_file = "tscal_ncore_p6_m3.csv"
    # csv_file = os.path.join(os.path.dirname(__file__), "tscal_ncore_p6_m3.csv")

    plot_histogram_selected_cores(
        csv_path=csv_file,
        time_kind="Wall",
        agg="mean",
        selected_cores=(56, 28, 10),
        title=None
    )