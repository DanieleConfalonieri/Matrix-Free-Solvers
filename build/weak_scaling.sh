#!/bin/bash

# Parameters setup
CSV_FILE="weak_scaling.csv"
EXEC_MF="./weak_scaling_free"
EXEC_MB="./weak_scaling_based"
EXEC_CALC_CORES="./calc_core"
MULTIGRID_OPTION="-mg"

MAX_P=6
MAX_LEVEL_MF=3
MAX_LEVEL_MB=3
DIMENSION=1

# Recupero output dei core
OUT_CORES=$(./$EXEC_CALC_CORES $DIMENSION)
CORES_ARRAY=()

# Popoliamo l'array dei core
for p in $(seq 1 $MAX_P); do
    CORE_VAL=$(echo "$OUT_CORES" | grep -oP "with \K[0-9]+(?= cores for dimension $p)")
    CORES_ARRAY+=("$CORE_VAL")
done

# Intestazione CSV
echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

# Funzione di estrazione per evitare duplicazione di codice (opzionale, ma consigliata)
# Qui sistemiamo i cicli principali

# --- Matrix-Free Benchmark ---
echo "=== Profiling Matrix-Free ==="
for i in $(seq 0 $((MAX_P-1))); do
    p=$((i+1))
    n_cores=${CORES_ARRAY[$i]}

    for L in $(seq 3 $MAX_LEVEL_MF); do
        echo "   Executing Matrix-Free (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun -n $n_cores $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

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

# --- Matrix-Based Benchmark ---
echo "=== Profiling Matrix-Based ==="
for i in $(seq 0 $((MAX_P-1))); do
    p=$((i+1))
    n_cores=${CORES_ARRAY[$i]}

    for L in $(seq 3 $MAX_LEVEL_MB); do
        echo "   Executing Matrix-Based (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun -n $n_cores $EXEC_MB 1 $L "$MULTIGRID_OPTION" $p)

        SETUP_SYSTEM_CPU=$(echo "$OUTPUT" | grep -oP 'Setup system\s*\(CPU\/wall\)\s\K[0-9.]+')
        SETUP_SYSTEM_WALL=$(echo "$OUTPUT" | grep -oP 'Setup system\s*\(CPU\/wall\)\s[0-9.]+\s*s\/\s*\K[0-9.]+')
        NUMBER_ELEMENTS=$(echo "$OUTPUT" | grep -oP 'Number of elements = \K[0-9]+')
        DOFS=$(echo "$OUTPUT" | grep -oP 'Number of degrees of freedom:\s*\K\d+')
        DOFS_CELL=$(echo "$OUTPUT" | grep -oP 'DoFs per cell\s*=\s\K[0-9]+')
        QPOINTS_CELL=$(echo "$OUTPUT" | grep -oP 'Quadrature points per cell\s*=\s\K[0-9]+')
        LINEAR_SYSTEM_CPU=$(echo "$OUTPUT" | grep -oP 'Solve linear system\s*\(CPU\/wall\)\s\K[0-9.]+')
        LINEAR_SYSTEM_WALL=$(echo "$OUTPUT" | grep -oP 'Solve linear system\s*\(CPU\/wall\)\s[0-9.]+\s*s\/\K[0-9.]+')
        ITERS=$(echo "$OUTPUT" | grep -oP 'Solved in\s*\K\d+')
        TIME_ITER=$(echo "$OUTPUT" | grep -oP 'Time per iter:\s*\K[0-9.]+')
        THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Throughput:\s*\K[0-9.]+\s*MDoFs/s')
        ASSEMBLE_CPU=$(echo "$OUTPUT" | grep -oP 'Assembly complete\s*\(CPU\/wall\)\s*\K[0-9.]+')
        ASSEMBLE_WALL=$(echo "$OUTPUT" | grep -oP 'Assembly complete\s*\(CPU\/wall\)\s*[0-9.]+s\/\K[0-9.]+')

        if [ ! -z "$SETUP_SYSTEM_CPU" ] && [ ! -z "$SETUP_SYSTEM_WALL" ] && [ ! -z "$NUMBER_ELEMENTS" ] && [ ! -z "$DOFS" ] && [ ! -z "$DOFS_CELL" ] && [ ! -z "$QPOINTS_CELL" ] && [ ! -z "$RHS_CPU" ] && [ ! -z "$RHS_WALL" ] && [ ! -z "$LINEAR_SYSTEM_CPU" ] && [ ! -z "$LINEAR_SYSTEM_WALL" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ] && [ ! -z "$THROUGHPUT" ]; then
            echo "$n_cores,MatrixBased,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$ASSEMBLE_CPU,$ASSEMBLE_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="