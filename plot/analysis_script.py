import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ==============================================================================
# 1. CONFIGURAZIONE STILE GRAFICI
# ==============================================================================
def setup_plot_style():
    """Configura uno stile professionale per i grafici."""
    # Cerchiamo di usare uno stile pulito, con fallback se non disponibile
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
        'lines.markersize': 8
    })

# ==============================================================================
# 2. ESTRAZIONE E PROCESSING
# ==============================================================================
def extract_and_process_data(filepath):
    """
    Carica il CSV, gestisce l'encoding e calcola le colonne dei totali.
    """
    if not os.path.exists(filepath):
        # Generiamo un errore critico se il file non c'è, ma gestito nel main
        raise FileNotFoundError(f"File '{filepath}' non trovato.")

    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    df = None
    
    # Tentativo di lettura con diversi encoding
    for enc in encodings:
        try:
            # Leggiamo tutto subito, se fallisce passa al prossimo encoding
            df_temp = pd.read_csv(filepath, sep=',', encoding=enc)
            
            # Controllo base: se ha meno di 2 colonne, probabilmente il separatore è sbagliato
            # o l'encoding ha rotto tutto.
            if len(df_temp.columns) > 1:
                print(f"-> File caricato con successo (encoding: {enc})")
                df = df_temp
                break 
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if df is None:
        raise ValueError("Impossibile determinare l'encoding corretto o leggere il file.")

    # --- CLEANING ---
    # Rimuoviamo spazi dai nomi delle colonne
    df.columns = df.columns.str.strip()
    
    # Colonne necessarie per il calcolo
    required_cols = [
        'SetupSystemCPU', 'RhsCpu', 'SolvingLinearSystemCPU',
        'SetupSystemWall', 'RhsWall', 'SolvingLinearSystemWall'
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Il CSV manca delle seguenti colonne necessarie: {missing}")

    # --- CALCOLO TOTALI ---
    # Somma CPU Time
    df['TotalTimeCPU'] = (
        df['SetupSystemCPU'] + 
        df['RhsCpu'] + 
        df['SolvingLinearSystemCPU']
    )
    
    # Somma Wall Time
    df['TotalTimeWall'] = (
        df['SetupSystemWall'] + 
        df['RhsWall'] + 
        df['SolvingLinearSystemWall']
    )

    print("-> Colonne 'TotalTimeCPU' e 'TotalTimeWall' calcolate.")
    return df

# ==============================================================================
# 3. FUNZIONI DI PLOTTING
# ==============================================================================

def plot_strong_scaling(df):
    """
    Grafico 1: Strong Scaling
    X=Cores, Y=TotalWall, Degree=6, Refinement=3
    Trace: MatrixFree vs MatrixBased (se presenti)
    """
    # Filtro richiesto dalla specifica
    subset = df[(df['Degree'] == 6) & (df['Refinement'] == 3)].copy()
    
    if subset.empty:
        print("Warning: Dati insufficienti per Strong Scaling (Degree=6, Ref=3). Salto grafico.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    solvers = subset['Solver'].unique()
    
    for solver in solvers:
        # Ordiniamo per Cores per avere una linea coerente
        data = subset[subset['Solver'] == solver].sort_values('Cores')
        ax.plot(data['Cores'], data['TotalTimeWall'], marker='o', label=solver)

    ax.set_title('Strong Scaling (Degree=6, Ref=3)', fontweight='bold')
    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Total Wall Time (s)')
    
    # Scala Log-Log per strong scaling
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Gestione ticks asse X (mostra i numeri interi dei core presenti)
    unique_cores = sorted(subset['Cores'].unique())
    ax.set_xticks(unique_cores)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_weak_scaling(df):
    """
    Grafico 2: Weak Scaling (Evoluzione per Degree)
    X=Cores, Y=TotalWall
    Traces: Diversi Degree (mostrando l'evoluzione)
    """
    # Nota: Assumiamo MatrixFree come riferimento principale se non specificato,
    # oppure plottiamo tutto. Qui plottiamo MatrixFree per pulizia, come spesso accade.
    solver_target = 'MatrixFree'
    subset = df[df['Solver'] == solver_target].copy()
    
    if subset.empty:
        # Fallback se non c'è MatrixFree, usiamo il primo disponibile
        if not df['Solver'].empty:
            solver_target = df['Solver'].iloc[0]
            subset = df[df['Solver'] == solver_target].copy()
        else:
            print("Warning: Dati vuoti per Weak Scaling.")
            return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    degrees = sorted(subset['Degree'].unique())
    
    # Mappa colori per i degree
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(degrees))]

    for i, deg in enumerate(degrees):
        data = subset[subset['Degree'] == deg].sort_values('Cores')
        if not data.empty:
            ax.plot(data['Cores'], data['TotalTimeWall'], 
                    marker='s', label=f'Degree {deg}', color=colors[i])

    ax.set_title(f'Scaling Evolution by Degree ({solver_target})', fontweight='bold')
    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Total Wall Time (s)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.legend(title="Polynomial Degree")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_problem_size_stacked(df):
    """
    Grafico 3: Istogrammi Stacked Side-by-Side (MatrixFree vs MatrixBased)
    X=Degree, Y=WallTime, Cores=56
    Componenti: Setup, Rhs, Solve
    """
    # Filtro Cores = 56
    subset = df[df['Cores'] == 56].sort_values('Degree')
    if subset.empty:
        print("Warning: Nessun dato trovato per Cores=56. Salto grafico Stacked.")
        return

    degrees = sorted(subset['Degree'].unique())
    solvers = sorted(subset['Solver'].unique()) # Es: ['MatrixBased', 'MatrixFree']
    
    # Componenti da impilare
    components = ['SetupSystemWall', 'RhsWall', 'SolvingLinearSystemWall']
    comp_labels = ['Setup', 'RHS', 'Solve']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Verde, Giallo, Rosso

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(degrees))
    width = 0.35  # Larghezza barre
    
    # Loop sui solver (MatrixFree vs MatrixBased) per posizionarli side-by-side
    for i, solver in enumerate(solvers):
        # Filtriamo i dati per questo solver
        solver_data = subset[subset['Solver'] == solver].set_index('Degree')
        
        # Reindicizziamo per assicurare che ci siano tutti i degree (riempie NaN con 0)
        solver_data = solver_data.reindex(degrees).fillna(0)
        
        # Calcolo posizione x (shift a sinistra o destra)
        # Se c'è 1 solver, offset=0. Se ce ne sono 2: -width/2 e +width/2
        if len(solvers) > 1:
            offset = (i - 0.5) * width * 1.1 # 1.1 per dare un piccolo spazio tra i gruppi
        else:
            offset = 0
            
        # Variabile per tracciare il fondo delle barre (per lo stacking)
        bottom = np.zeros(len(degrees))
        
        # Loop sulle componenti (Setup, Rhs, Solve)
        for j, comp in enumerate(components):
            vals = solver_data[comp].values
            
            # Label solo per il primo solver per evitare duplicati in legenda
            # Ma qui la legenda è tricky perché abbiamo colori (componenti) e posizioni (solver).
            # Non mettiamo label qui, la costruiamo custom dopo.
            
            # Tratteggio per distinguere i solver visivamente (opzionale)
            hatch = '//' if i == 1 else ''
            edge_color = 'white'
            
            ax.bar(x + offset, vals, width, bottom=bottom, 
                   color=colors[j], edgecolor=edge_color, hatch=hatch, alpha=0.9)
            
            bottom += vals

    # Decorazioni
    ax.set_title('Time Breakdown by Degree (Cores=56)', fontweight='bold')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Wall Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)
    
    # --- LEGENDA PERSONALIZZATA ---
    from matplotlib.patches import Patch
    
    # Legenda Colori (Componenti)
    legend_elements = [Patch(facecolor=colors[j], label=comp_labels[j]) for j in range(len(colors))]
    
    # Legenda Solver (Pattern)
    legend_elements.append(Patch(facecolor='gray', label=solvers[0], alpha=0.5))
    if len(solvers) > 1:
        legend_elements.append(Patch(facecolor='gray', hatch='//', label=solvers[1], alpha=0.5))

    ax.legend(handles=legend_elements, title="Components & Solvers")
    
    plt.tight_layout()
    plt.show()

def plot_problem_size_lines(df):
    """
    Grafico 4: Line Plot Total Time vs Degree
    X=Degree, Y=TotalWallTime, Cores=56
    """
    subset = df[df['Cores'] == 56].sort_values('Degree')
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    solvers = subset['Solver'].unique()

    for solver in solvers:
        data = subset[subset['Solver'] == solver].sort_values('Degree')
        ax.plot(data['Degree'], data['TotalTimeWall'], 
                marker='D', markersize=8, linewidth=2.5, label=solver)

    ax.set_title('Total Time vs Degree (Cores=56)', fontweight='bold')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Total Wall Time (s)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    setup_plot_style()
    
    # Imposta il nome del tuo file CSV qui
    # Nota: Assicurati che il file sia nella stessa cartella o fornisci il path completo
    filename = 'tscal_ncores_p6_m3.csv' 

    try:
        print(f"--- Inizio Analisi: {filename} ---")
        
        # 1. Estrazione e Calcolo
        df_processed = extract_and_process_data(filename)
        
        print(f"Dati caricati: {len(df_processed)} righe.")
        # Anteprima dati
        print(df_processed[['Cores', 'Solver', 'Degree', 'TotalTimeWall']].head())

        # 2. Generazione Grafici
        print("\nGenerazione Grafico 1: Strong Scaling...")
        plot_strong_scaling(df_processed)
        
        print("Generazione Grafico 2: Weak Scaling...")
        plot_weak_scaling(df_processed)
        
        print("Generazione Grafico 3: Stacked Bar Analysis...")
        plot_problem_size_stacked(df_processed)
        
        print("Generazione Grafico 4: Total Time Lines...")
        plot_problem_size_lines(df_processed)
        
        print("\n--- Elaborazione completata con successo. ---")

    except Exception as e:
        print(f"\nERRORE CRITICO:\n{e}")
        import traceback
        traceback.print_exc()