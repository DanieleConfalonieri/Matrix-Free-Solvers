import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
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

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
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

# ==============================================================================
# 3. LETTURA DATI E PRE-PROCESSING
# ==============================================================================
def extract_and_process_data(filepath_or_buffer):
    # Legge il file (o il buffer se passi direttamente i dati come stringa)
    df = pd.read_csv(filepath_or_buffer, sep=',')
    df.columns = df.columns.str.strip()
    
    # Normalizza i nomi dei solver
    if 'Solver' in df.columns:
        df['Solver_norm'] = df['Solver'].apply(normalize_solver_name)

    # Calcola il TotalTimeWall se non esiste
    wall_parts = ['SetupSystemWall', 'RhsOrAssembleWall', 'SolvingLinearSystemWall']
    if 'TotalTimeWall' not in df.columns:
        if all(c in df.columns for c in wall_parts):
            df['TotalTimeWall'] = df[wall_parts].sum(axis=1)
        else:
            print("Attenzione: Mancano alcune colonne per calcolare il TotalTimeWall.")

    # Pulisce la colonna del throughput (rimuove ' MDoFs/s' e converte in float)
    if 'Throughput(MDoFs/s)' in df.columns:
        df['Throughput_Num'] = df['Throughput(MDoFs/s)'].astype(str).str.replace(r'[a-zA-Z/ ]', '', regex=True)
        df['Throughput_Num'] = pd.to_numeric(df['Throughput_Num'], errors='coerce')

    # Converti colonne chiave in numeriche
    for col in ['Cores', 'Degree', 'Refinement', 'TotalTimeWall']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ==============================================================================
# 4. PLOT: WEAK SCALING TIME
# ==============================================================================
def plot_time_weak_scaling(df, refinement=6):
    """
    Nello weak scaling, idealmente il tempo totale rimane costante 
    all'aumentare dei core (e della dimensione del problema).
    """
    subset = df[(df['Refinement'] == refinement)].copy()
    subset = subset.dropna(subset=['Cores', 'TotalTimeWall'])
    
    if subset.empty:
        print("Nessun dato disponibile per il plot del Time.")
        return

    cores_all = sorted(subset['Cores'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    solvers = subset['Solver_norm'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver_norm'] == solver].sort_values('Cores')
        if not data.empty:
            ax.plot(data['Cores'], data['TotalTimeWall'], marker='o', color=colors[i], label=f"{solver}")

    ax.set_title(f"Weak Scaling: Total Wall Time (Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_xscale('log')
    # ax.set_yscale('log') # Scommenta se i tempi variano di molti ordini di grandezza
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. PLOT: WEAK SCALING EFFICIENCY
# ==============================================================================
def plot_efficiency_weak_scaling(df, refinement=6):
    """
    E_weak = T_base / T_p.
    T_base è il tempo con il minor numero di core per quello specifico solver.
    """
    subset = df[(df['Refinement'] == refinement)].copy()
    subset = subset.dropna(subset=['Cores', 'TotalTimeWall'])

    if subset.empty:
        return

    cores_all = sorted(subset['Cores'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    # Linea ideale E_w = 1
    ax.plot(cores_all, np.ones(len(cores_all)), linestyle='--', marker='x', color='black', alpha=0.7, label='Ideal E_w=1')

    solvers = subset['Solver_norm'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver_norm'] == solver].sort_values('Cores')
        if not data.empty:
            # Trova il tempo base (minor numero di core)
            base_core = data['Cores'].min()
            t_base = data[data['Cores'] == base_core]['TotalTimeWall'].values[0]
            
            # Calcola efficienza: T_base / T_p
            data['Efficiency'] = t_base / data['TotalTimeWall']
            ax.plot(data['Cores'], data['Efficiency'], marker='o', color=colors[i], label=f"{solver} (Base p={base_core})")

    ax.set_title(f"Weak Scaling: Efficiency (Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Efficiency ($T_{base} / T_p$)")
    ax.set_xscale('log')
    ax.set_ylim(bottom=0.0, top=1.1)
    apply_fixed_log_xticks(ax, cores_all)

    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 6. PLOT: THROUGHPUT
# ==============================================================================
def plot_throughput_weak_scaling(df, refinement=6):
    """
    Nello weak scaling ideale, il throughput totale (MDoFs/s) scala linearmente
    con il numero di core.
    """
    subset = df[(df['Refinement'] == refinement)].copy()
    subset = subset.dropna(subset=['Cores', 'Throughput_Num'])

    if subset.empty:
        print("Nessun dato di Throughput disponibile.")
        return

    cores_all = sorted(subset['Cores'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    solvers = subset['Solver_norm'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(solvers))))

    for i, solver in enumerate(solvers):
        data = subset[subset['Solver_norm'] == solver].sort_values('Cores')
        if not data.empty:
            ax.plot(data['Cores'], data['Throughput_Num'], marker='s', color=colors[i], label=f"{solver}")

            # Plot ideal scaling for throughput based on the smallest core count
            base_core = data['Cores'].min()
            base_throughput = data[data['Cores'] == base_core]['Throughput_Num'].values[0]
            ideal_throughput = (data['Cores'] / base_core) * base_throughput
            ax.plot(data['Cores'], ideal_throughput, linestyle=':', color=colors[i], alpha=0.6, label=f"{solver} (Ideal)")

    ax.set_title(f"Weak Scaling: Throughput (Ref={refinement})", fontweight='bold')
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Throughput (MDoFs/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
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

    # Per comodità, uso i dati che hai fornito simulando la lettura di un CSV.
    # Nel tuo caso reale, sostituisci 'csv_data' con il percorso del tuo file, 
    # es: df = extract_and_process_data('mio_file_weak_scaling.csv')
    csv_data = """Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsOrAssembleCpu,RhsOrAssembleWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)
2,MatrixFree,1,6,0.021145,0.0223487,4096,4225,4,4,0.000912,0.000916451,0.019933,0.0208083,13,0.00160064,2.63957 MDoFs/s
16,MatrixFree,2,6,0.021965,0.0243217,4096,16641,9,9,0,0.000967913,0.025868,0.0287534,17,0.00169138,9.83873 MDoFs/s
54,MatrixFree,3,6,0.032323,0.0389158,4096,37249,16,16,0.001156,0.00116418,0.042667,0.0427415,19,0.00224955,16.5584 MDoFs/s
2,MatrixBased,1,6,1.99681,2.06153,262144,274625,8,8,0.894241,0.898021,2.54783,2.25099,30,0.0750331,3.66005 MDoFs/s
16,MatrixBased,2,6,2.01744,2.07934,262144,2146689,27,27,4.71417,4.73644,227.507,210.35,50,4.207,0.510266 MDoFs/s
"""
    filepath = "weak_scaling.csv"
    try:
        df = extract_and_process_data(filepath)

        print("\n=== Analisi Weak Scaling ===")
        print(df[['Cores', 'Solver_norm', 'Degree', 'TotalTimeWall', 'Throughput_Num']])

        plot_time_weak_scaling(df, refinement=6)
        plot_efficiency_weak_scaling(df, refinement=6)
        plot_throughput_weak_scaling(df, refinement=6)

    except Exception as e:
        print(f"ERRORE: {e}")