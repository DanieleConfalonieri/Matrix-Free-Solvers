import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# ==============================================================================
# 1. CONFIGURAZIONE STILE
# ==============================================================================
def setup_plot_style():
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot')

    # (opzionale ma spesso utile) evita MathText sui formatter automatici
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
        'axes.formatter.use_mathtext': False,  # aiuta a evitare 3×10^1
    })

# ==============================================================================
# 2. NORMALIZZAZIONE SOLVER
# ==============================================================================
def normalize_solver_name(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    s = s.replace('_', ' ').replace('-', ' ')
    s = re.sub(r'\s+', ' ', s)

    if s in ('matrixfree', 'matrix free'):
        return 'matrix free'
    if s in ('matrixbased', 'matrix based'):
        return 'matrix based'

    if 'matrix' in s and 'free' in s:
        return 'matrix free'
    if 'matrix' in s and 'based' in s:
        return 'matrix based'
    return s

def add_solver_norm(df: pd.DataFrame, col='Solver', newcol='Solver_norm'):
    df = df.copy()
    df[newcol] = df[col].apply(normalize_solver_name) if col in df.columns else ''
    return df

# ==============================================================================
# 3. LETTURA + TOTAL TIME
# ==============================================================================
def extract_and_process_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' non trovato.")

    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    df = None

    for enc in encodings:
        try:
            df_temp = pd.read_csv(filepath, sep=',', encoding=enc)
            if len(df_temp.columns) > 1:
                print(f"-> File caricato: {filepath} (encoding: {enc})")
                df = df_temp
                break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        raise ValueError(f"Impossibile leggere '{filepath}' con gli encoding provati.")

    df.columns = df.columns.str.strip()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    wall_parts = ['SetupSystemWall', 'RhsWall', 'SolvingLinearSystemWall']
    if 'TotalTimeWall' not in df.columns:
        if all(c in df.columns for c in wall_parts):
            df['TotalTimeWall'] = df['SetupSystemWall'] + df['RhsWall'] + df['SolvingLinearSystemWall']
            print(f"-> {filepath}: calcolata 'TotalTimeWall' dai componenti Wall.")
        else:
            raise ValueError(f"{filepath}: manca 'TotalTimeWall' e mancano componenti Wall {wall_parts}.")

    for col in ['Cores', 'Degree', 'Refinement', 'TotalTimeWall']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ==============================================================================
# 4. STIMA T1 (ultima spiaggia)
# ==============================================================================
def estimate_t1_from_tp(df_case):
    d = df_case.dropna(subset=['Cores', 'TotalTimeWall']).copy()
    d = d[(d['Cores'] > 0) & (d['TotalTimeWall'] > 0)]
    if len(d) < 2:
        return np.nan
    x = np.log(d['Cores'].values.astype(float))
    y = np.log(d['TotalTimeWall'].values.astype(float))
    b, a = np.polyfit(x, y, 1)  # y = b*x + a
    t1 = float(np.exp(a))
    return t1 if np.isfinite(t1) and t1 > 0 else np.nan

# ==============================================================================
# 5. BASELINE ROBUSTA
# ==============================================================================
def add_onecore_baseline_robust(df_multi, df_one,
                               keys=('Solver', 'Degree', 'Refinement'),
                               allow_degree_ref_fallback=True,
                               allow_from_multi_1core=True,
                               allow_estimate=True):
    dfm = add_solver_norm(df_multi, 'Solver', 'Solver_norm')
    dfo = add_solver_norm(df_one, 'Solver', 'Solver_norm')

    keys_eff = [k if k != 'Solver' else 'Solver_norm' for k in keys]

    if 'TotalTimeWall' not in dfm.columns or 'TotalTimeWall' not in dfo.columns:
        raise ValueError("Serve 'TotalTimeWall' in entrambi i dataframe.")

    dfm['T1_Wall'] = np.nan
    dfm['T1_Source'] = 'missing'

    # A) one-core esatto
    dfo_1 = dfo.copy()
    if 'Cores' in dfo_1.columns:
        dfo_1 = dfo_1[dfo_1['Cores'] == 1]

    t1_exact = (dfo_1.dropna(subset=['TotalTimeWall'])
                .groupby(keys_eff)['TotalTimeWall']
                .median())
    idx = pd.MultiIndex.from_frame(dfm[keys_eff])
    dfm['T1_Wall'] = idx.map(t1_exact).to_numpy()
    dfm.loc[dfm['T1_Wall'].notna(), 'T1_Source'] = 'onecore_exact'

    # B) fallback Degree/Refinement
    if allow_degree_ref_fallback and ('Degree' in dfm.columns) and ('Refinement' in dfm.columns):
        miss = dfm['T1_Wall'].isna()
        if miss.any():
            t1_degref = (dfo_1.dropna(subset=['TotalTimeWall'])
                         .groupby(['Degree', 'Refinement'])['TotalTimeWall']
                         .median())
            idx2 = pd.MultiIndex.from_arrays([dfm.loc[miss, 'Degree'], dfm.loc[miss, 'Refinement']])
            dfm.loc[miss, 'T1_Wall'] = idx2.map(t1_degref).to_numpy()
            dfm.loc[miss & dfm['T1_Wall'].notna(), 'T1_Source'] = 'onecore_degref'

    # C) fallback dal multi-core a 1 core
    if allow_from_multi_1core and 'Cores' in dfm.columns:
        miss = dfm['T1_Wall'].isna()
        if miss.any():
            dfm_1 = dfm[dfm['Cores'] == 1].copy()
            if not dfm_1.empty:
                t1_from_multi = (dfm_1.dropna(subset=['TotalTimeWall'])
                                 .groupby(keys_eff)['TotalTimeWall']
                                 .median())
                idx3 = pd.MultiIndex.from_frame(dfm.loc[miss, keys_eff])
                dfm.loc[miss, 'T1_Wall'] = idx3.map(t1_from_multi).to_numpy()
                dfm.loc[miss & dfm['T1_Wall'].notna(), 'T1_Source'] = 'multi_1core'

    # D) stima T1
    if allow_estimate:
        miss = dfm['T1_Wall'].isna()
        if miss.any():
            case_cols = [c for c in ['Solver_norm', 'Degree', 'Refinement'] if c in dfm.columns]
            for case_key, _ in dfm[miss].groupby(case_cols):
                selector = np.ones(len(dfm), dtype=bool)
                if isinstance(case_key, tuple):
                    for col, val in zip(case_cols, case_key):
                        selector &= (dfm[col] == val)
                else:
                    selector &= (dfm[case_cols[0]] == case_key)

                t1 = estimate_t1_from_tp(dfm[selector])
                if np.isfinite(t1):
                    dfm.loc[selector & dfm['T1_Wall'].isna(), 'T1_Wall'] = t1
                    dfm.loc[selector & (dfm['T1_Source'] == 'missing'), 'T1_Source'] = 'estimated'

    print("\n=== Baseline T1_Wall report ===")
    print(dfm['T1_Source'].value_counts(dropna=False))
    return dfm

# ==============================================================================
# helper: tick asse X su log scale con etichette fissate (NO 3×10¹)
# ==============================================================================
def apply_fixed_log_xticks(ax, ticks):
    ticks = sorted({float(t) for t in ticks})
    labels = []
    for t in ticks:
        if float(t).is_integer():
            labels.append(str(int(t)))
        else:
            labels.append(str(t))

    # forza locator/formatter (sovrascrive LogFormatterMathtext)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))

    # opzionale: niente tick minori (spesso più pulito nei plot HPC)
    ax.xaxis.set_minor_locator(NullLocator())

# ==============================================================================
# 6. PLOT SPEEDUP
# ==============================================================================
def plot_speedup_strong_scaling(df, degree=6, refinement=3):
    base = df[(df['Degree'] == degree) & (df['Refinement'] == refinement)].copy()
    base = base.dropna(subset=['Cores'])
    base = base[base['Cores'] > 0]

    cores_all = sorted(base['Cores'].unique())
    if len(cores_all) == 0:
        print(f"Warning: nessun core disponibile (Degree={degree}, Ref={refinement}).")
        return

    subset = base.dropna(subset=['TotalTimeWall', 'T1_Wall']).copy()
    subset = subset[(subset['TotalTimeWall'] > 0) & (subset['T1_Wall'] > 0)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cores_all, cores_all, linestyle='--', marker='x', alpha=0.7, label='Ideal S(p)=p')

    if subset.empty:
        ax.set_title(f"Speedup Strong Scaling (Degree={degree}, Ref={refinement})", fontweight='bold')
        ax.set_xlabel("Number of Cores")
        ax.set_ylabel("Speedup (T1 / Tp)")
        ax.set_xscale('log')
        ax.set_yscale('log')

        apply_fixed_log_xticks(ax, cores_all)

        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()
        print("Nota: non ci sono punti speedup (manca T1_Wall/TotalTimeWall), ma la linea ideale è stata tracciata.")
        return

    subset['Speedup'] = subset['T1_Wall'] / subset['TotalTimeWall']

    solvers = subset['Solver'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver'] == solver].sort_values('Cores')
        if data.empty:
            continue
        ax.plot(data['Cores'], data['Speedup'], marker='o', color=colors[i], label=solver)

    ax.set_title(f"Speedup Strong Scaling (Degree={degree}, Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Speedup (T1 / Tp)")
    ax.set_xscale('log')
    ax.set_yscale('log')

    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 7. PLOT EFFICIENCY
# ==============================================================================
def plot_efficiency_strong_scaling(df, degree=6, refinement=3):
    base = df[(df['Degree'] == degree) & (df['Refinement'] == refinement)].copy()
    base = base.dropna(subset=['Cores'])
    base = base[base['Cores'] > 0]

    cores_all = sorted(base['Cores'].unique())
    if len(cores_all) == 0:
        print(f"Warning: nessun core disponibile (Degree={degree}, Ref={refinement}).")
        return

    subset = base.dropna(subset=['TotalTimeWall', 'T1_Wall', 'Cores']).copy()
    subset = subset[(subset['TotalTimeWall'] > 0) & (subset['T1_Wall'] > 0) & (subset['Cores'] > 0)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cores_all, np.ones(len(cores_all)), linestyle='--', marker='x', alpha=0.7, label='Ideal E(p)=1')

    if subset.empty:
        ax.set_title(f"Efficiency Strong Scaling (Degree={degree}, Ref={refinement})", fontweight='bold')
        ax.set_xlabel("Number of Cores")
        ax.set_ylabel("Efficiency (T1 / (p*Tp))")
        ax.set_xscale('log')

        apply_fixed_log_xticks(ax, cores_all)

        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()
        print("Nota: non ci sono punti efficiency (manca T1_Wall/TotalTimeWall), ma la linea ideale è stata tracciata.")
        return

    subset['Efficiency'] = subset['T1_Wall'] / (subset['Cores'] * subset['TotalTimeWall'])

    solvers = subset['Solver'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver'] == solver].sort_values('Cores')
        if data.empty:
            continue
        ax.plot(data['Cores'], data['Efficiency'], marker='o', color=colors[i], label=solver)

    ax.set_title(f"Efficiency Strong Scaling (Degree={degree}, Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Efficiency (T1 / (p*Tp))")
    ax.set_xscale('log')

    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    setup_plot_style()

    filename_multi = "tscal_ncore_p6_m3.csv"
    filename_one = "onecoreMatrixProfiling.csv"

    try:
        df_multi = extract_and_process_data(filename_multi)
        df_one = extract_and_process_data(filename_one)

        df_multi = add_onecore_baseline_robust(df_multi, df_one)

        print("\nCores disponibili (Degree=6 Ref=3):",
              sorted(df_multi[(df_multi['Degree'] == 6) & (df_multi['Refinement'] == 3)]
                     ['Cores'].dropna().unique()))

        plot_speedup_strong_scaling(df_multi, degree=6, refinement=3)
        plot_efficiency_strong_scaling(df_multi, degree=6, refinement=3)

    except Exception as e:
        print(f"ERRORE: {e}")
