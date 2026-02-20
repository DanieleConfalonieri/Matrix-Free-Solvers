#!/bin/bash

CSV_FILE="weak_scaling.csv"
EXEC_MF="./weak_scaling_free"
EXEC_MB="./weak_scaling_based"
EXEC_CALC_CORES="./calc_core"
MULTIGRID_OPTION="-mg"

MAX_P=6
MAX_LEVEL_MF=6
MAX_LEVEL_MB=6
DIMENSION=2

echo "--- Esecuzione $EXEC_CALC_CORES per determinare i core ---"
OUT_CORES=$(./$EXEC_CALC_CORES $DIMENSION)
CORES_ARRAY=()

for p in $(seq 1 $MAX_P); do
    VAL=$(echo "$OUT_CORES" | grep -oP "with \K[0-9]+(?= cores for dimension $DIMENSION and p = $p)" | head -n 1 | tr -dc '0-9')
    FINAL_VAL=${VAL:-1}
    CORES_ARRAY+=("$FINAL_VAL")
    echo " > Per grado p=$p salvati $FINAL_VAL core"
done
echo "--------------------------------------------------------"

echo "Cores,Solver,Degree,Refinement,SetupSystemCPU,SetupSystemWall,NumberOfElements,DoFs,DoFsPerCell,QPointsPerCell,RhsOrAssembleCpu,RhsOrAssembleWall,SolvingLinearSystemCPU,SolvingLinearSystemWall,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

echo "=== Profiling Matrix-Free ==="
for i in "${!CORES_ARRAY[@]}"; do
    p=$((i+1))
    n_cores="${CORES_ARRAY[$i]}"

    for L in $(seq 6 $MAX_LEVEL_MF); do
        echo "   Executing Matrix-Free (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun --oversubscribe -n "$n_cores" $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)

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

        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ]; then
            echo "$n_cores,MatrixFree,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$RHS_CPU,$RHS_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
        fi
    done
done

echo "=== Profiling Matrix-Based ==="
for i in "${!CORES_ARRAY[@]}"; do
    p=$((i+1))
    n_cores="${CORES_ARRAY[$i]}"

    for L in $(seq 6 $MAX_LEVEL_MB); do
        echo "   Executing Matrix-Based (Cores=$n_cores, p=$p, Level=$L)..."

        OUTPUT=$(mpirun --oversubscribe -n "$n_cores" $EXEC_MB 1 $L "$MULTIGRID_OPTION" $p)

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

        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ]; then
            echo "$n_cores,MatrixBased,$p,$L,$SETUP_SYSTEM_CPU,$SETUP_SYSTEM_WALL,$NUMBER_ELEMENTS,$DOFS,$DOFS_CELL,$QPOINTS_CELL,$ASSEMBLE_CPU,$ASSEMBLE_WALL,$LINEAR_SYSTEM_CPU,$LINEAR_SYSTEM_WALL,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per Cores=$n_cores, p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Results saved in $CSV_FILE ==="