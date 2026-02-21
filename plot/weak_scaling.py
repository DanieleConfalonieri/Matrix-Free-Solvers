import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
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

    wall_parts = ['SetupSystemWall', 'RhsOrAssembleWall', 'SolvingLinearSystemWall']
    if 'TotalTimeWall' not in df.columns:
        if all(c in df.columns for c in wall_parts):
            df['TotalTimeWall'] = df[wall_parts].sum(axis=1)
        else:
            print("Warning: Missing columns to compute TotalTimeWall.")

    if 'Throughput(MDoFs/s)' in df.columns:
        df['Throughput_Num'] = df['Throughput(MDoFs/s)'].astype(str).str.replace(r'[a-zA-Z/ ]', '', regex=True)
        df['Throughput_Num'] = pd.to_numeric(df['Throughput_Num'], errors='coerce')

    for col in ['Cores', 'Degree', 'Refinement', 'TotalTimeWall']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_time_strong_scaling_per_p(df, refinement=6):
    subset = df[df['Refinement'] == refinement].copy()
    subset = subset.dropna(subset=['Cores', 'TotalTimeWall', 'Degree'])

    if subset.empty:
        print("No data available for the Time plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    cores_all = sorted(subset['Cores'].unique())
    degrees = sorted(subset['Degree'].unique())

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(degrees))))

    for i, p in enumerate(degrees):
        data_p = subset[subset['Degree'] == p].sort_values('Cores')
        if not data_p.empty:
            ax.plot(data_p['Cores'], data_p['TotalTimeWall'], marker='o',
                    color=colors[i], label=f"Real (p={int(p)})")

            base_core = data_p['Cores'].min()
            base_time = data_p[data_p['Cores'] == base_core]['TotalTimeWall'].values[0]

            ideal_cores = np.array([c for c in cores_all if c >= base_core])
            if len(ideal_cores) > 0:
                ideal_time = base_time * (base_core / ideal_cores)
                ax.plot(ideal_cores, ideal_time, linestyle='--', color=colors[i],
                        alpha=0.6, label=f"Ideal Strong (p={int(p)})")

    ax.set_title(f"Time vs Cores (Strong Scaling traces per $p$) - Ref={refinement}", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Total Wall Time (s)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig('time_vs_cores_strong_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_efficiency_weak_scaling(df, refinement=6):
    subset = df[(df['Refinement'] == refinement)].copy()
    subset = subset.dropna(subset=['Cores', 'TotalTimeWall'])

    if subset.empty:
        return

    cores_all = sorted(subset['Cores'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cores_all, np.ones(len(cores_all)), linestyle='--', marker='x', color='black', alpha=0.7,
            label='Ideal E_w=1')

    solvers = subset['Solver_norm'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver_norm'] == solver].sort_values('Cores')
        if not data.empty:
            base_core = data['Cores'].min()
            t_base = data[data['Cores'] == base_core]['TotalTimeWall'].values[0]

            data['Efficiency'] = t_base / data['TotalTimeWall']
            ax.plot(data['Cores'], data['Efficiency'], marker='o', color=colors[i],
                    label=f"{solver} (Base p={base_core})")

    ax.set_title(f"Weak Scaling: Efficiency (Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Efficiency ($T_{base} / T_p$)")
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.0, top=1.1)
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('efficiency_weak_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_throughput_weak_scaling(df, refinement=6):
    subset = df[(df['Refinement'] == refinement)].copy()
    subset = subset.dropna(subset=['Cores', 'Throughput_Num'])

    if subset.empty:
        print("No Throughput data available.")
        return

    cores_all = sorted(subset['Cores'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    solvers = subset['Solver_norm'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver_norm'] == solver].sort_values('Cores')
        if not data.empty:
            ax.plot(data['Cores'], data['Throughput_Num'], marker='s', color=colors[i], label=f"{solver}")

            base_core = data['Cores'].min()
            base_throughput = data[data['Cores'] == base_core]['Throughput_Num'].values[0]
            ideal_throughput = (data['Cores'] / base_core) * base_throughput
            ax.plot(data['Cores'], ideal_throughput, linestyle=':', color=colors[i], alpha=0.6,
                    label=f"{solver} (Ideal)")

    ax.set_title(f"Weak Scaling: Throughput (Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Throughput (MDoFs/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('throughput_weak_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    setup_plot_style()

    filepath = "weak_scaling.csv"
    try:
        df = extract_and_process_data(filepath)

        print("\n=== Weak Scaling Analysis (MatrixFree Only) ===")
        print(df[['Cores', 'Degree', 'TotalTimeWall', 'Throughput_Num']])

        plot_time_strong_scaling_per_p(df, refinement=6)

        plot_efficiency_weak_scaling(df, refinement=6)

        plot_throughput_weak_scaling(df, refinement=6)

    except Exception as e:
        print(f"ERROR: {e}")