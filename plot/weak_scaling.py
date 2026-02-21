import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def setup_plot_style():
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot')

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.formatter.use_mathtext': False,
    })


def normalize_solver_name(s: str) -> str:
    if pd.isna(s):
        return ''
    s = str(s).strip().lower()
    if 'matrix' in s and 'free' in s:
        return 'Matrix Free'
    if 'matrix' in s and 'based' in s:
        return 'Matrix Based'
    return s.title()


def apply_fixed_log_xticks(ax, ticks):
    ticks = sorted({float(t) for t in ticks})
    labels = [str(int(t)) if float(t).is_integer() else str(t) for t in ticks]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())


def extract_and_process_data(filepath_or_buffer):
    df = pd.read_csv(filepath_or_buffer, sep=',')
    df.columns = df.columns.str.strip()

    if 'Solver' in df.columns:
        df['Solver_norm'] = df['Solver'].apply(normalize_solver_name)
        df = df[df['Solver_norm'] == 'Matrix Free'].copy()

    if 'TotalTimeWall' not in df.columns:
        rhs_col = 'RhsOrAssembleWall' if 'RhsOrAssembleWall' in df.columns else 'RhsWall'
        wall_parts = ['SetupSystemWall', rhs_col, 'SolvingLinearSystemWall']

        if all(c in df.columns for c in wall_parts):
            df['TotalTimeWall'] = df[wall_parts].sum(axis=1)
        else:
            print(f"Warning: Missing columns to compute TotalTimeWall. Expected: {wall_parts}")

    if 'Throughput(MDoFs/s)' in df.columns:
        df['Throughput_Num'] = df['Throughput(MDoFs/s)'].astype(str).str.replace(r'[a-zA-Z/ ]', '', regex=True)
        df['Throughput_Num'] = pd.to_numeric(df['Throughput_Num'], errors='coerce')

    for col in ['Cores', 'Degree', 'Refinement', 'TotalTimeWall']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_time_strong_scaling_per_p(df, refinement):
    subset = df[df['Refinement'] == refinement].copy()
    subset = subset.dropna(subset=['Cores', 'TotalTimeWall', 'Degree'])

    # --- FILTERS FOR PLOT CLEANUP ---
    # 1. Discretization every 2 (even cores only)
    subset = subset[subset['Cores'] % 2 == 0]

    # 2. For p = 1, start from 8 cores (discard cases where Degree==1 and Cores<8)
    subset = subset[~((subset['Degree'] == 1) & (subset['Cores'] < 2))]
    # ----------------------------------

    if subset.empty:
        print(f"No data available for the Time plot with Refinement={refinement} after filtering.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    cores_all = sorted(subset['Cores'].unique())
    degrees = sorted(subset['Degree'].unique())

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(degrees))))

    for i, p in enumerate(degrees):
        data_p = subset[subset['Degree'] == p].sort_values('Cores')
        if not data_p.empty:
            # base_core will be 8 for p=1, 16 for p=2, and 54 for p=3
            base_core = data_p['Cores'].min()

            ax.plot(data_p['Cores'], data_p['TotalTimeWall'], marker='o',
                    color=colors[i], label=f"Real Strong (p={int(p)})")

            base_time = data_p[data_p['Cores'] == base_core]['TotalTimeWall'].values[0]
            ideal_cores = np.array([c for c in cores_all if c >= base_core])

            if len(ideal_cores) > 0:
                ideal_time = base_time * (base_core / ideal_cores)
                ax.plot(ideal_cores, ideal_time, linestyle='--', color=colors[i],
                        alpha=0.6, label=f"Ideal Strong (p={int(p)}, base p={int(base_core)})")

    ax.set_title(f"Strong Scaling: Time vs Cores per Degree $p$ (Refinement={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Total Wall Time (s)")

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Now the ticks on the x-axis will show only the filtered cores (even, starting from 8)
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f'time_vs_cores_strong_scaling_ref{int(refinement)}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    setup_plot_style()

    filepath = "weak_scaling.csv"
    try:
        df = extract_and_process_data(filepath)

        if df.empty:
            print("The DataFrame is empty. Check the CSV file.")
        else:
            print("\n=== Strong Scaling Analysis per Polynomial Degree ===")

            unique_refinements = df['Refinement'].dropna().unique()

            for ref in unique_refinements:
                print(f"\nGenerating plots for Refinement = {int(ref)}...")
                plot_time_strong_scaling_per_p(df, refinement=ref)

    except Exception as e:
        print(f"ERROR: {e}")