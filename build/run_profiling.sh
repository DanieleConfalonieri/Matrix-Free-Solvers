#!/bin/bash

# Parameters setup
CSV_FILE="hpc_benchmark_results.csv"
EXEC_MF="./profiling_free"
EXEC_MB="./profiling_based"
MULTIGRID_OPTION="-mg"

MAX_LEVEL_MF=3
MAX_LEVEL_MB=3

# Definizione lista core: 1, e poi da 2 a 56 con passo 2
CORES_LIST="$(seq 8 4 34) $(seq 36 10 56)"

# Intestazione CSV (Aggiunta colonna Cores all'inizio)
echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

# --- Matrix-Free Benchmark ---
echo "=== Profiling Matrix-Free ==="
for n_cores in $CORES_LIST; do
    echo "--- Running with $n_cores cores ---"
    for p in {1..6}; do
        for L in $(seq 3 $MAX_LEVEL_MF); do
            echo "   Executing Matrix-Free (Cores=$n_cores, p=$p, Level=$L)..."

            OUTPUT=$(mpirun -n $n_cores $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

            # Estrazione dati (Regex invariate)
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
            #OUTPUT_RES_CPU=$(echo "$OUTPUT" | grep -oP 'Output')
            #OUTPUT_RES_WALL=$(echo "$OUTPUT" | grep -oP 'results\s*\(CPU\/wall\)\s*[0-9.]+s\/\K[0-9.]+')

            if [ ! -z "$SETUP_SYSTEM_CPU" ] && [ ! -z "$SETUP_SYSTEM_WALL" ] && [ ! -z "$NUMBER_ELEMENTS" ] && [ ! -z "$DOFS" ] && [ ! -z "$DOFS_CELL" ] && [ ! -z "$QPOINTS_CELL" ] && [ ! -z "$RHS_CPU" ] && [ ! -z "$RHS_WALL" ] && [ ! -z "$LINEAR_SYSTEM_CPU" ] && [ ! -z "$LINEAR_SYSTEM_WALL" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ] && [ ! -z "$THROUGHPUT" ]; then
                echo "$n_cores,MatrixFree,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$RHS_CPU,$RHS_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
            else
                echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
            fi
        done
    done
done

# --- Matrix-Based Benchmark ---

CORES_LIST="$(seq 8 4 34) $(seq 36 10 56)"

echo "=== Profiling Matrix-Based ==="
for n_cores in $CORES_LIST; do
    echo "--- Running with $n_cores cores ---"
    for p in {1..6}; do
        for L in $(seq 3 $MAX_LEVEL_MB); do
            echo "   Executing Matrix-Based (Cores=$n_cores, p=$p, Level=$L)..."

            OUTPUT=$(mpirun -n $n_cores $EXEC_MB 1 $L "$MULTIGRID_OPTION" $p)

            # Estrazione dati specifica per Matrix-Based
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

	        if [ ! -z "$SETUP_SYSTEM_CPU" ] && [ ! -z "$SETUP_SYSTEM_WALL" ] && [ ! -z "$NUMBER_ELEMENTS" ] && [ ! -z "$DOFS" ] && [ ! -z "$DOFS_CELL" ] && [ ! -z "$QPOINTS_CELL" ] && [ ! -z "$ASSEMBLE_CPU" ] && [ ! -z "$ASSEMBLE_WALL" ] && [ ! -z "$LINEAR_SYSTEM_CPU" ] && [ ! -z "$LINEAR_SYSTEM_WALL" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ] && [ ! -z "$THROUGHPUT" ]; then
                echo "$n_cores,MatrixBased,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$ASSEMBLE_CPU,$ASSEMBLE_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
            else
                echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
            fi
        done
    done
done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="