#!/bin/bash

# Parameters setup
CSV_FILE="hpc_benchmark_results.csv"
EXEC_MF="./profiling_free"
EXEC_MB="./profiling_based"
MULTIGRID_OPTION="-mg"

L=6


echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsCpu,RhsWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

get_cores_list() {
    local degree=$1
    if [ "$degree" -eq 1 ]; then
        seq 2 2 56
    elif [ "$degree" -eq 2 ]; then
        seq 16 2 56
    elif [ "$degree" -eq 3 ]; then
        seq 54 2 56
    fi
}

echo "=== Profiling Matrix-Free ==="
for p in {1..3}; do
    CORES_LIST=$(get_cores_list $p)
    echo "--- Testing Matrix-Free with p=$p ---"
    
    for n_cores in $CORES_LIST; do
        echo "   Executing Matrix-Free (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun -n $n_cores $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

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

echo "=== Profiling Matrix-Based ==="
for p in {1..3}; do
    CORES_LIST=$(get_cores_list $p)
    echo "--- Testing Matrix-Based with p=$p ---"
    
    for n_cores in $CORES_LIST; do
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

        if [ ! -z "$SETUP_SYSTEM_CPU" ] && [ ! -z "$SETUP_SYSTEM_WALL" ] && [ ! -z "$NUMBER_ELEMENTS" ] && [ ! -z "$DOFS" ] && [ ! -z "$DOFS_CELL" ] && [ ! -z "$QPOINTS_CELL" ] && [ ! -z "$ASSEMBLE_CPU" ] && [ ! -z "$ASSEMBLE_WALL" ] && [ ! -z "$LINEAR_SYSTEM_CPU" ] && [ ! -z "$LINEAR_SYSTEM_WALL" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ] && [ ! -z "$THROUGHPUT" ]; then
            echo "$n_cores,MatrixBased,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$ASSEMBLE_CPU,$ASSEMBLE_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="