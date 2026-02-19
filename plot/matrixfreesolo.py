import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Carica i dati ---
df = pd.read_csv("solomatrixfree.csv")

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
df = df.dropna(subset=["DoFs", "SolvingLinearSystemWall"])
df = df[(df["DoFs"] > 0) & (df["SolvingLinearSystemWall"] > 0)]

# --- 4) Plot log-log ---
plt.figure()

# una curva per solver (modifica se vuoi altre combinazioni)
for solver, g in df.groupby("Solver"):
    g = g.sort_values("DoFs")
    plt.loglog(
        g["DoFs"],
        g["SolvingLinearSystemWall"],
        marker="o",
        linewidth=1.5,
        label=str(solver)
    )

plt.xlabel("DoFs")
plt.ylabel("SolvingLinearSystemWall (s)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
