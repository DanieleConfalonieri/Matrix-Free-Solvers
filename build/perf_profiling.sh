#!/bin/bash

PARADIGM="free"
EXEC="./profiling_"$PARADIGM 
CORES=8                  
CSV_FILE="hardware_scalability_${PARADIGM}.csv"

PERF_EVENTS="cycles,instructions,cache-misses,L1-dcache-load-misses"

echo "Degree,Level,Avg_Cycles,Avg_Instructions,AVG_IPC,Avg_LLC_Misses,Avg_L1_Misses,Avg_MaxRSS_KB,Total_Mem_GB" > $CSV_FILE

echo "=== Avvio Profiling Hardware + RAM (No-BC) su $CORES core ==="

for p in {2..4}; do
    for L in {4..6}; do
        
        echo " -> Profiling p=$p, Level=$L..."

        rm -f stats_rank_*.txt mem_rank_*.txt

        mpirun -n $CORES bash -c "/usr/bin/time -f '%M' -o mem_rank_\${OMPI_COMM_WORLD_RANK}.txt perf stat -x, -o stats_rank_\${OMPI_COMM_WORLD_RANK}.txt -e $PERF_EVENTS $EXEC 1 $L -mg $p" > /dev/null 2>&1

        TOT_CYCLES=0
        TOT_INSTR=0
        TOT_LLC=0
        TOT_L1=0
        TOT_MEM_KB=0

        for i in $(seq 0 $((CORES - 1))); do
            FILE_PERF="stats_rank_${i}.txt"
            FILE_MEM="mem_rank_${i}.txt"
            
            # Parsing PERF
            if [ -f "$FILE_PERF" ]; then
                C=$(grep "cycles" $FILE_PERF | cut -d',' -f1)
                I=$(grep "instructions" $FILE_PERF | cut -d',' -f1)
                LLC=$(grep "cache-misses" $FILE_PERF | cut -d',' -f1)
                L1=$(grep "L1-dcache" $FILE_PERF | cut -d',' -f1)

                if ! [[ "$C" =~ ^[0-9]+$ ]]; then C=0; fi
                if ! [[ "$I" =~ ^[0-9]+$ ]]; then I=0; fi
                if ! [[ "$LLC" =~ ^[0-9]+$ ]]; then LLC=0; fi
                if ! [[ "$L1" =~ ^[0-9]+$ ]]; then L1=0; fi
                
                TOT_CYCLES=$((TOT_CYCLES + C))
                TOT_INSTR=$((TOT_INSTR + I))
                TOT_LLC=$((TOT_LLC + LLC))
                TOT_L1=$((TOT_L1 + L1))
            fi

            # Parsing MEMORY 
            if [ -f "$FILE_MEM" ]; then
                KB=$(cat $FILE_MEM)
                if ! [[ "$KB" =~ ^[0-9]+$ ]]; then KB=0; fi
                TOT_MEM_KB=$((TOT_MEM_KB + KB))
            fi
        done
        
        AVG_CYCLES=$((TOT_CYCLES / CORES))
        AVG_INSTR=$((TOT_INSTR / CORES))
        AVG_LLC=$((TOT_LLC / CORES))
        AVG_L1=$((TOT_L1 / CORES))
        
        # Average Memory per core (KB)
        AVG_MEM_KB=$((TOT_MEM_KB / CORES))

        if [ "$TOT_CYCLES" -gt 0 ]; then
            IPC=$(awk -v i="$TOT_INSTR" -v c="$TOT_CYCLES" 'BEGIN {printf "%.3f", i/c}')
        else
            IPC="0"
        fi

        # Total RAM usage (GB)
        # (KB total) / 1024 / 1024
        TOTAL_MEM_GB=$(awk -v k="$TOT_MEM_KB" 'BEGIN {printf "%.4f", k/1024/1024}')

        echo "    Result: IPC=$IPC, LLC_Miss=$AVG_LLC, MemTotal=${TOTAL_MEM_GB} GB"

        echo "$p,$L,$AVG_CYCLES,$AVG_INSTR,$IPC,$AVG_LLC,$AVG_L1,$AVG_MEM_KB,$TOTAL_MEM_GB" >> $CSV_FILE
    done
done

rm -f stats_rank_*.txt mem_rank_*.txt
echo "=== Succesful! Data in $CSV_FILE ==="