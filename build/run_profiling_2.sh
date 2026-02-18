#!/bin/bash

# Parameters setup
CSV_FILE="hpc_benchmark_results.csv"
EXEC_MF="./profiling_free"
EXEC_MB="./profiling_based"
MULTIGRID_OPTION="-mg"

MAX_LEVEL_MF=4
N_CORES=56

# Definizione lista core: 1, e poi da 2 a 56 con passo 2"

# Intestazione CSV (Aggiunta colonna Cores all'inizio)
echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s),OutputResultCpu,OutputResultWall" > $CSV_FILE

# --- Matrix-Free Benchmark ---
echo "=== Profiling Matrix-Free ==="
    for p in {1..12}; do
        for L in $(seq 4 $MAX_LEVEL_MF); do
            echo "   Executing Matrix-Free (Cores=$N_CORES, p=$p, Level=$L)..."

            OUTPUT=$(mpirun -n $N_CORES $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

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
            OUTPUT_RES_CPU=$(echo "$OUTPUT" | grep -oP 'Output')
            OUTPUT_RES_WALL=$(echo "$OUTPUT" | grep -oP 'results\s*\(CPU\/wall\)\s*[0-9.]+s\/\K[0-9.]+')

            if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ]; then
                # Aggiunta n_cores come primo campo
                echo "$N_CORES,MatrixFree,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$RHS_CPU,$RHS_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT,$OUTPUT_RES_CPU,$OUTPUT_RES_WALL" >> $CSV_FILE
            else
                echo "   [!] Fallito o OOM per Cores=$N_CORES, p=$p, Level=$L"
            fi
        done
    done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="