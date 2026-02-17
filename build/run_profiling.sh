#!/bin/bash

# Parameters setup 
CORES=4
CSV_FILE="hpc_benchmark_results.csv"

# Executables name
EXEC_MF="./profiling_free"
EXEC_MB="./profiling_based" 

# Multigrid option
MULTIGRID_OPTION="-mg" # use -mg for multigrid, other for no multigrid

# MatrixBased will likely run out of memory for higher levels, so we may want
# to set different max levels for the two benchmarks
MAX_LEVEL_MF=5
MAX_LEVEL_MB=4 

echo "Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s),OutputResultCpu,OutputResultWall" > $CSV_FILE

# Matrix-Free Benchmark
echo "=== Profiling Matrix-Free ==="
for p in {1..1}; do
    for L in $(seq 3 $MAX_LEVEL_MF); do
        echo "   Executing Matrix-Free (p=$p, Level=$L)..."

        # mpirun (parameters: profiling=1, level=L, mg=MULTIGRID_OPTION, degree=p)
        OUTPUT=$(mpirun -n $CORES $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

        # Extract data using grep e regex
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

        # If extraction successful, append to CSV, otherwise log failure
        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ]; then
            echo "MatrixFree,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$RHS_CPU,$RHS_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT,$OUTPUT_RES_CPU,$OUTPUT_RES_WALL" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per p=$p, Level=$L"
        fi
    done
done

# Matrix-Based Benchmark
echo "=== Profiling Matrix-Based ==="

for p in {1..6}; do
    for L in $(seq 3 $MAX_LEVEL_MB); do
        echo "   Executing Matrix-Based (p=$p, Level=$L)..."

        OUTPUT=$(mpirun -n $CORES $EXEC_MB 1 $L "$MULTIGRID_OPTION" $p)

        
        DOFS=$(echo "$OUTPUT" | grep -oP 'Number of degrees of freedom:\s*\K\d+')
        LINEAR_SYSTEM=$(echo "$OUTPUT" | grep -oP 'Solve linear system\s*\(CPU\/wall\)\s*\K[0-9.]+')
        ITERS=$(echo "$OUTPUT" | grep -oP 'Solved in\s*\K\d+')
        TIME_ITER=$(echo "$OUTPUT" | grep -oP 'Time per iter:\s*\K[0-9.]+')
        THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Throughput:\s*\K[0-9.]+\s*MDoFs/s')
        

        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ]; then
            echo "MatrixBased,$p,$L,$NUMBER_ELEMENTS,$DOFS,$LINEAR_SYSTEM,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Risultats saved in $CSV_FILE ==="