#!/bin/bash


PROFILING_FLAG=1      
MG_FLAG="-mg"         
DEGREE=3              


CORES=(1 4 16)
REFINEMENTS=(7 8 9)


echo "=================================================================="
echo " Inizio Analisi di Weak Scaling (2D)"
echo " Grado Polinomiale fisso (p) : $DEGREE"
echo " Precondizionatore           : $MG_FLAG"
echo "=================================================================="

echo -e "\n---> Esecuzione Solutore MATRIX-FREE <---"
for i in "${!CORES[@]}"; do
    n_cores=${CORES[$i]}
    ref_level=${REFINEMENTS[$i]}

    echo "[$(date +'%H:%M:%S')] Lancio run: $n_cores Core | Refinement $ref_level"
    
    mpirun -n $n_cores ./profiling_free $PROFILING_FLAG $ref_level $MG_FLAG $DEGREE
    
    echo "Completato run con $n_cores core."
    echo "-------------------------------------------------"
done

echo -e "\n---> Esecuzione Solutore MATRIX-BASED <---"
for i in "${!CORES[@]}"; do
    n_cores=${CORES[$i]}
    ref_level=${REFINEMENTS[$i]}

    echo "[$(date +'%H:%M:%S')] Lancio run: $n_cores Core | Refinement $ref_level"
    
    mpirun -n $n_cores ./profiling_based $PROFILING_FLAG $ref_level $MG_FLAG $DEGREE
    
    echo "Completato run con $n_cores core."
    echo "-------------------------------------------------"
done

echo "=================================================================="
echo " Analisi Weak Scaling completata!"
echo "=================================================================="