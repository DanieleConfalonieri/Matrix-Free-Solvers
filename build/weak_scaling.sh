#!/bin/bash

# Parameters setup
CSV_FILE="weak_scaling.csv"
EXEC_MF="./weak_scaling_free"
EXEC_CALC_CORES="./calc_core"
MULTIGRID_OPTION="-mg"

MAX_P=3
L=6 # Mesh refinement fisso a 6
DIMENSION=2
MAX_CORES=56

# --- Calcolo dinamico dei core base ---
echo "--- Esecuzione $EXEC_CALC_CORES per determinare i core base ---"
OUT_CORES=$(./$EXEC_CALC_CORES $DIMENSION)
CORES_BASE_ARRAY=()

for p in $(seq 1 $MAX_P); do
    VAL=$(echo "$OUT_CORES" | grep -oP "with \K[0-9]+(?= cores for dimension $DIMENSION and p = $p)" | head -n 1 | tr -dc '0-9')
    FINAL_VAL=${VAL:-1}
    CORES_BASE_ARRAY+=("$FINAL_VAL")
    echo " > Per grado p=$p, il numero di core base e' $FINAL_VAL"
done
echo "--------------------------------------------------------"

# Intestazione CSV
echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

# --- Matrix-Free Benchmark ---
echo "=== Profiling Matrix-Free ==="

# Itero sui gradi polinomiali
for i in "${!CORES_BASE_ARRAY[@]}"; do
    p=$((i+1))
    base_cores="${CORES_BASE_ARRAY[$i]}"
    
    # Costruisco la lista delle core da testare per questo p
    # Da base_cores a MAX_CORES, con passo 2 (o passo 1 se modifichi)
    # Se base_cores > MAX_CORES, seq potrebbe comportarsi in modo anomalo, quindi facciamo un controllo
    if [ "$base_cores" -le "$MAX_CORES" ]; then
        CORES_LIST=$(seq "$base_cores" 2 "$MAX_CORES")
    else
        echo "Attenzione: i core base per p=$p ($base_cores) superano MAX_CORES ($MAX_CORES). Verrà testato solo con $base_cores core."
        CORES_LIST="$base_cores"
    fi

    echo "--- Testing Matrix-Free with p=$p (Da $base_cores a $MAX_CORES core) ---"
    
    # Itero sulle core
    for n_cores in $CORES_LIST; do
        echo "   Executing Matrix-Free (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun -n "$n_cores" $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

        # Estrazione dati
        SETUP_SYSTEM_CPU=$(echo "$OUTPUT" | grep -oP 'Setup matrix-free system\s*\(CPU\/wall\)\s\K[0-9.]+')
        SETUP_SYSTEM_WALL=$(echo "$OUTPUT" | grep -oP 'Setup matrix-free system\s*\(CPU\/wall\)\s[0-9.]+\s*s\/\s*\K[0-9.]+')
        NUMBER_ELEMENTS=$(echo "$OUTPUT" | grep -oP 'Number of elements = \K[0-9]+')
        DOFS=$(echo "$OUTPUT" | grep -oP 'Number of degrees of freedom:\s*\K\d+')
        DOFS_CELL=$(echo "$OUTPUT" | grep -oP 'DoFs per cell\s*=\s\K[0-9]+')
        QPOINTS_CELL=$(echo "$OUTPUT" | grep -oP 'Quadrature points per cell\s*=\s\K[0-9]+')
        RHS_CPU=$(echo "$OUTPUT" | grep -oP 'right hand side\s*\(CPU\/wall\)\s*\K[0-9.]+')
        RHS_WALL=$(echo "$OUTPUT" | grep -oP 'right hand side\s*\(CPU\/wall\)\s*[0-9.]+s\/\K[0-9.]+')
        LINEAR_SYSTEM_CPU=$(echo "$OUTPUT" | grep -oP 'Solve linear system\s*\(CPU\/wall\)\s\K[0-9.]+')
        LINEAR_SYSTEM_WALL=$(echo "$OUTPUT" | grep -oP 'Solve linear system\s*\(CPU\/wall\)\s[0-9.]+\s*s\/\K[0-9.]+')
        ITERS=$(echo "$OUTPUT" | grep -oP 'Solved in\s*\K\d+')
        TIME_ITER=$(echo "$OUTPUT" | grep -oP 'Time per iter:\s*\K[0-9.]+')
        THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Throughput:\s*\K[0-9.]+\s*MDoFs/s')

        if [ ! -z "$SETUP_SYSTEM_CPU" ] && [ ! -z "$SETUP_SYSTEM_WALL" ] && [ ! -z "$NUMBER_ELEMENTS" ] && [ ! -z "$DOFS" ] && [ ! -z "$DOFS_CELL" ] && [ ! -z "$QPOINTS_CELL" ] && [ ! -z "$RHS_CPU" ] && [ ! -z "$RHS_WALL" ] && [ ! -z "$LINEAR_SYSTEM_CPU" ] && [ ! -z "$LINEAR_SYSTEM_WALL" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ] && [ ! -z "$THROUGHPUT" ]; then
            echo "$n_cores,MatrixFree,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$RHS_CPU,$RHS_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="