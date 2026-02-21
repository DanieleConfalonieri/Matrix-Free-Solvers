#!/bin/bash


PROFILING_FLAG=1      
MG_FLAG="-mg"         
DEGREE=3              


CORES=(1 4 16)
REFINEMENTS=(7 8 9)


echo "=================================================================="
echo " Starting Weak Scaling Analysis (2D)"
echo " Fixed Polynomial Degree (p) : $DEGREE"
echo " Preconditioner              : $MG_FLAG"
echo "=================================================================="

echo -e "\n---> Running MATRIX-FREE Solver <---"
for i in "${!CORES[@]}"; do
    n_cores=${CORES[$i]}
    ref_level=${REFINEMENTS[$i]}

    echo "[$(date +'%H:%M:%S')] Running: $n_cores Core | Refinement $ref_level"
    
    mpirun -n $n_cores ./profiling_free $PROFILING_FLAG $ref_level $MG_FLAG $DEGREE
    
    echo "Completed run with $n_cores cores."
    echo "-------------------------------------------------"
done

echo -e "\n---> Running MATRIX-BASED Solver <---"
for i in "${!CORES[@]}"; do
    n_cores=${CORES[$i]}
    ref_level=${REFINEMENTS[$i]}

    echo "[$(date +'%H:%M:%S')] Running: $n_cores Core | Refinement $ref_level"
    
    mpirun -n $n_cores ./profiling_based $PROFILING_FLAG $ref_level $MG_FLAG $DEGREE
    
    echo "Completed run with $n_cores cores."
    echo "-------------------------------------------------"
done

echo "=================================================================="
echo " Weak Scaling Analysis completed!"
echo "=================================================================="