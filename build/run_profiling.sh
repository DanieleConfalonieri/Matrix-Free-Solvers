#!/bin/bash

# Parameters setup 
CORES=4
CSV_FILE="hpc_benchmark_results.csv"

# Executables name
EXEC_MF="./matrix_free"
EXEC_MB="./matrix_based" 

# Multigrid option
MULTIGRID_OPTION="-mg" # use -mg for multigrid, other for no multigrid

# MatrixBased will likely run out of memory for higher levels, so we may want
# to set different max levels for the two benchmarks
MAX_LEVEL_MF=5
MAX_LEVEL_MB=5 

echo "Solver,Degree,Refinement,DoFs,Iterations,TimePerIter(s),Throughput(MDoFs/s)" > $CSV_FILE

# Matrix-Free Benchmark
echo "=== Profiling Matrix-Free ==="
for p in {1..6}; do
    for L in $(seq 3 $MAX_LEVEL_MF); do
        echo "   Executing Matrix-Free (p=$p, Level=$L)..."
        
        # mpirun (parameters: profiling=1, level=L, mg=MULTIGRID_OPTION, degree=p)
        OUTPUT=$(mpirun -n $CORES $EXEC_MF 1 $L "$MULTIGRID_OPTION" $p)
        
        # Extract data using grep e regex 
        DOFS=$(echo "$OUTPUT" | grep -oP 'Number of degrees of freedom:\s*\K\d+')
        ITERS=$(echo "$OUTPUT" | grep -oP 'Solved in\s*\K\d+')
        TIME_ITER=$(echo "$OUTPUT" | grep -oP 'Time per iter:\s*\K[0-9.]+')
        THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Throughput:\s*\K[0-9.]+\s*MDoFs/s')
        
        # If extraction successful, append to CSV, otherwise log failure
        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ]; then
            echo "MatrixFree,$p,$L,$DOFS,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
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
        ITERS=$(echo "$OUTPUT" | grep -oP 'Solved in\s*\K\d+')
        TIME_ITER=$(echo "$OUTPUT" | grep -oP 'Time per iter:\s*\K[0-9.]+')
        THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Throughput:\s*\K[0-9.]+\s*MDoFs/s')
        
        if [ ! -z "$DOFS" ] && [ ! -z "$THROUGHPUT" ] && [ ! -z "$ITERS" ] && [ ! -z "$TIME_ITER" ]; then
            echo "MatrixBased,$p,$L,$DOFS,$ITERS,$TIME_ITER,$THROUGHPUT" >> $CSV_FILE
        else
            echo "   [!] Fallito o OOM per p=$p, Level=$L"
        fi
    done
done

echo "=== Benchmark completed! Risultats saved in $CSV_FILE ==="