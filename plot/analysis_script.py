import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ==================================================================
# 1. PLOT STYLE CONFIGURATION
# ==================================================================
def setup_plot_style():
    """Configure a professional plotting style."""
    # Try to use a clean style, with fallbacks if unavailable
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

# ==================================================================
# 2. EXTRACTION AND PROCESSING
# ==================================================================
def extract_and_process_data(filepath):
    """Load the CSV, handle encoding, and compute total columns."""
    if not os.path.exists(filepath):
        # Raise a critical error if the file is missing; handled in main
        raise FileNotFoundError(f"File '{filepath}' not found.")

    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    df = None
    
    # Attempt reading with multiple encodings
    for enc in encodings:
        try:
            # Read immediately; on failure try the next encoding
            df_temp = pd.read_csv(filepath, sep=',', encoding=enc)
            
            # Basic check: if less than 2 columns, separator may be wrong
            # or the encoding may have corrupted the file.
            if len(df_temp.columns) > 1:
                print(f"-> File successfully loaded (encoding: {enc})")
                df = df_temp
                break 
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if df is None:
        raise ValueError("Impossibile determinare l'encoding corretto o leggere il file.")

    # --- CLEANING ---
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Columns required for calculations
    required_cols = [
        'SetupSystemCPU', 'RhsCpu', 'SolvingLinearSystemCPU',
        'SetupSystemWall', 'RhsWall', 'SolvingLinearSystemWall'
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing the following required columns: {missing}")

    # --- TOTALS CALCULATION ---
    # Sum CPU Time
    df['TotalTimeCPU'] = (
        df['SetupSystemCPU'] + 
        df['RhsCpu'] + 
        df['SolvingLinearSystemCPU']
    )
    
    # Sum Wall Time
    df['TotalTimeWall'] = (
        df['SetupSystemWall'] + 
        df['RhsWall'] + 
        df['SolvingLinearSystemWall']
    )

    print("-> Columns 'TotalTimeCPU' and 'TotalTimeWall' computed.")
    return df

# ==================================================================
# 3. PLOTTING FUNCTIONS
# ==================================================================

def plot_strong_scaling(df):
    """
    Grafico 1: Strong Scaling
    X=Cores, Y=TotalWall, Degree=6, Refinement=3
    Trace: MatrixFree vs MatrixBased (se presenti)
    """
    # Filter required by specification
    subset = df[(df['Degree'] == 6) & (df['Refinement'] == 3)].copy()
    
    if subset.empty:
        print("Warning: Insufficient data for Strong Scaling (Degree=6, Ref=3). Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    solvers = subset['Solver'].unique()
    
    for solver in solvers:
        # Sort by Cores to have a coherent line
        data = subset[subset['Solver'] == solver].sort_values('Cores')
        ax.plot(data['Cores'], data['TotalTimeWall'], marker='o', label=solver)

    ax.set_title('Strong Scaling (Degree=6, Ref=3)', fontweight='bold')
    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Total Wall Time (s)')
    
    # Use log-log scale for strong scaling
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Handle X-axis ticks (show integer core counts present)
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
    # Note: Prefer MatrixFree as primary reference if present; otherwise plot available solver.
    # Here we plot MatrixFree for clarity, which is commonly used.
    solver_target = 'MatrixFree'
    subset = df[df['Solver'] == solver_target].copy()
    
    if subset.empty:
        # Fallback: if MatrixFree is missing, use the first available solver
        if not df['Solver'].empty:
            solver_target = df['Solver'].iloc[0]
            subset = df[df['Solver'] == solver_target].copy()
        else:
            print("Warning: Empty data for Weak Scaling.")
            return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    degrees = sorted(subset['Degree'].unique())
    
    # Color map for degrees
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
    Plot 3: Side-by-side stacked bar charts (MatrixFree vs MatrixBased)
    X=Degree, Y=WallTime, Cores=56
    Components: Setup, Rhs, Solve
    """
    # Filter for Cores = 56
    subset = df[df['Cores'] == 56].sort_values('Degree')
    if subset.empty:
        print("Warning: No data found for Cores=56. Skipping Stacked plot.")
        return

    degrees = sorted(subset['Degree'].unique())
    solvers = sorted(subset['Solver'].unique())  # e.g. ['MatrixBased', 'MatrixFree']

    # Components to stack
    components = ['SetupSystemWall', 'RhsWall', 'SolvingLinearSystemWall']
    comp_labels = ['Setup', 'RHS', 'Solve']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(degrees))
    width = 0.35  # Bar width

    # Loop over solvers to position them side-by-side
    for i, solver in enumerate(solvers):
        # Filter data for this solver
        solver_data = subset[subset['Solver'] == solver].set_index('Degree')

        # Reindex to ensure all degrees are present (fill NaN with 0)
        solver_data = solver_data.reindex(degrees).fillna(0)

        # Compute x position (shift left or right)
        # If 1 solver -> offset=0; if 2 solvers -> -width/2 and +width/2
        if len(solvers) > 1:
            offset = (i - 0.5) * width * 1.1  # 1.1 to leave a small gap between groups
        else:
            offset = 0

        # Track bottom of bars for stacking
        bottom = np.zeros(len(degrees))

        # Loop over components (Setup, Rhs, Solve)
        for j, comp in enumerate(components):
            vals = solver_data[comp].values

            # Hatching to visually distinguish solvers (optional)
            hatch = '//' if i == 1 else ''
            edge_color = 'white'

            ax.bar(x + offset, vals, width, bottom=bottom,
                   color=colors[j], edgecolor=edge_color, hatch=hatch, alpha=0.9)

            bottom += vals

    # Decorations
    ax.set_title('Time Breakdown by Degree (Cores=56)', fontweight='bold')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Wall Time (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)

    # --- CUSTOM LEGEND ---
    from matplotlib.patches import Patch

    # Color legend (Components)
    legend_elements = [Patch(facecolor=colors[j], label=comp_labels[j]) for j in range(len(colors))]

    # Solver legend (Pattern)
    legend_elements.append(Patch(facecolor='gray', label=solvers[0], alpha=0.5))
    if len(solvers) > 1:
        legend_elements.append(Patch(facecolor='gray', hatch='//', label=solvers[1], alpha=0.5))

    ax.legend(handles=legend_elements, title="Components & Solvers")

    plt.tight_layout()
    plt.show()

def plot_problem_size_lines(df):
    """Plot 4: Line Plot Total Time vs Degree
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
    
    # Set your CSV filename here
    # Note: Ensure the file is in the same folder or provide the full path
    filename = 'tscal_ncores_p6_m3.csv' 

    try:
        print(f"--- Starting Analysis: {filename} ---")
        
        # 1. Extraction and Calculation
        df_processed = extract_and_process_data(filename)
        
        print(f"Loaded data: {len(df_processed)} rows.")
        # Data preview
        print(df_processed[['Cores', 'Solver', 'Degree', 'TotalTimeWall']].head())

        # 2. Generate plots
        print("\nGenerating Plot 1: Strong Scaling...")
        plot_strong_scaling(df_processed)
        
        print("Generating Plot 2: Weak Scaling...")
        plot_weak_scaling(df_processed)
        
        print("Generating Plot 3: Stacked Bar Analysis...")
        plot_problem_size_stacked(df_processed)
        
        print("Generating Plot 4: Total Time Lines...")
        plot_problem_size_lines(df_processed)
        
        print("\n--- Processing completed successfully. ---")

    except Exception as e:
        print(f"\nCRITICAL ERROR:\n{e}")
        import traceback
        traceback.print_exc()