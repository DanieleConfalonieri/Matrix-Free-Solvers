#!/bin/bash

PARADIGM="free"
EXEC="./profiling_"$PARADIGM 
CORES=8                  
CSV_FILE="hardware_scalability_${PARADIGM}.csv"

# cache-misses     = LLC Misses
# cache-references = LLC Accesses (Loads)
PERF_EVENTS="cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads"

echo "Degree,Level,Avg_IPC,L1_Miss_Rate_Perc,LLC_Miss_Rate_Perc,Avg_Cycles,Avg_L1_Misses,Avg_LLC_Misses,Total_Mem_GB" > $CSV_FILE

echo "=== Profiling started on $CORES cores ==="

for L in 4; do
    for p in {2..6}; do
        
        echo " -> Profiling p=$p, Level=$L..."

        rm -f stats_rank_*.txt mem_rank_*.txt

        # --bind-to core -> no mutltithreading
        mpirun -n $CORES --bind-to core \
        bash -c "/usr/bin/time -f '%M' -o mem_rank_\${OMPI_COMM_WORLD_RANK}.txt perf stat -x, -o stats_rank_\${OMPI_COMM_WORLD_RANK}.txt -e $PERF_EVENTS $EXEC 1 $L -mg $p" > /dev/null 2>&1
        
        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "    [!] CRASH DETECTED (Exit Code: $EXIT_CODE). Skipping analysis."
            echo "$p,$L,0,0,0,0,0,CRASH" >> $CSV_FILE
            continue
        fi

        TOT_CYCLES=0
        TOT_INSTR=0
        TOT_LLC_MISS=0
        TOT_LLC_REFS=0
        TOT_L1_MISS=0
        TOT_L1_LOADS=0
        TOT_MEM_KB=0

        for i in $(seq 0 $((CORES - 1))); do
            FILE_PERF="stats_rank_${i}.txt"
            FILE_MEM="mem_rank_${i}.txt"
            
            # Parsing PERF
            if [ -f "$FILE_PERF" ]; then
                C=$(grep "cycles" $FILE_PERF | cut -d',' -f1)
                I=$(grep "instructions" $FILE_PERF | cut -d',' -f1)
                LLC_M=$(grep "cache-misses" $FILE_PERF | cut -d',' -f1)
                LLC_R=$(grep "cache-references" $FILE_PERF | cut -d',' -f1)
                L1_M=$(grep "L1-dcache-load-misses" $FILE_PERF | cut -d',' -f1)
                L1_L=$(grep "L1-dcache-loads" $FILE_PERF | cut -d',' -f1)

                [[ ! "$C" =~ ^[0-9]+$ ]] && C=0
                [[ ! "$I" =~ ^[0-9]+$ ]] && I=0
                [[ ! "$LLC_M" =~ ^[0-9]+$ ]] && LLC_M=0
                [[ ! "$LLC_R" =~ ^[0-9]+$ ]] && LLC_R=0
                [[ ! "$L1_M" =~ ^[0-9]+$ ]] && L1_M=0
                [[ ! "$L1_L" =~ ^[0-9]+$ ]] && L1_L=0
                
                TOT_CYCLES=$((TOT_CYCLES + C))
                TOT_INSTR=$((TOT_INSTR + I))
                TOT_LLC_MISS=$((TOT_LLC_MISS + LLC_M))
                TOT_LLC_REFS=$((TOT_LLC_REFS + LLC_R))
                TOT_L1_MISS=$((TOT_L1_MISS + L1_M))
                TOT_L1_LOADS=$((TOT_L1_LOADS + L1_L))
            fi

            # Parsing MEMORY 
            if [ -f "$FILE_MEM" ]; then
                KB=$(cat $FILE_MEM)
                [[ ! "$KB" =~ ^[0-9]+$ ]] && KB=0
                TOT_MEM_KB=$((TOT_MEM_KB + KB))
            fi
        done
        
        
        AVG_CYCLES=$((TOT_CYCLES / CORES))
        AVG_L1_MISS=$((TOT_L1_MISS / CORES))
        AVG_LLC_MISS=$((TOT_LLC_MISS / CORES))

        # IPC
        if [ "$TOT_CYCLES" -gt 0 ]; then
            IPC=$(awk -v i="$TOT_INSTR" -v c="$TOT_CYCLES" 'BEGIN {printf "%.3f", i/c}')
        else
            IPC="0"
        fi

        # L1 Miss Rate (%)
        if [ "$TOT_L1_LOADS" -gt 0 ]; then
            L1_RATE=$(awk -v m="$TOT_L1_MISS" -v l="$TOT_L1_LOADS" 'BEGIN {printf "%.2f", (m/l)*100}')
        else
            L1_RATE="0.00"
        fi

        # LLC Miss Rate (%)
        if [ "$TOT_LLC_REFS" -gt 0 ]; then
            LLC_RATE=$(awk -v m="$TOT_LLC_MISS" -v r="$TOT_LLC_REFS" 'BEGIN {printf "%.2f", (m/r)*100}')
        else
            LLC_RATE="0.00"
        fi

        TOTAL_MEM_GB=$(awk -v k="$TOT_MEM_KB" 'BEGIN {printf "%.4f", k/1024/1024}')

        echo "    Result: IPC=$IPC | L1 Miss=$L1_RATE% | LLC Miss=$LLC_RATE% | Mem=${TOTAL_MEM_GB} GB"

        echo "$p,$L,$IPC,$L1_RATE,$LLC_RATE,$AVG_CYCLES,$AVG_L1_MISS,$AVG_LLC_MISS,$TOTAL_MEM_GB" >> $CSV_FILE
    done
done

rm -f stats_rank_*.txt mem_rank_*.txt
echo "=== Profiling Completed. Data saved in $CSV_FILE ==="