import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Carica i dati ---
df = pd.read_csv("MatrixFree_p12.csv")

# --- 2) Conversione colonne numeriche ---
num_cols = [
    "DoFs",
    "SolvingLinearSystemWall",
    "Cores",
    "Degree",
    "Refinement",
    "Iterations"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --- 3) Filtra righe valide ---
df = df.dropna(subset=["DoFs", "SolvingLinearSystemWall", "Refinement"])
df = df[(df["DoFs"] > 0) & (df["SolvingLinearSystemWall"] > 0)]

# --- 4) Plot log-log: un grafico per ogni Refinement ---
for ref, df_ref in df.groupby("Refinement"):

    plt.figure()

    # una curva per solver dentro ogni refinement
    for solver, g in df_ref.groupby("Solver"):
        g = g.sort_values("DoFs")

        plt.loglog(
            g["DoFs"],
            g["SolvingLinearSystemWall"],
            marker="o",
            linewidth=1.5,
            label=str(solver)
        )

    plt.title(f"Refinement = {ref}")
    plt.xlabel("DoFs")
    plt.ylabel("SolvingLinearSystemWall (s)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

plt.show()