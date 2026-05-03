#!/bin/bash
#BSUB -J bc_perf
#BSUB -W 03:00
#BSUB -q normal      
#BSUB -n 40
#BSUB -R "span[ptile=20]"
#BSUB -m "polus-c3-ib polus-c4-ib"
#BSUB -gpu "num=2:mode=shared"
#BSUB -o perf_%J.out
#BSUB -e perf_%J.err
cd ~/Brandes-algorithm-cuda/2023-PLGP-BC
module load SpectrumMPI

REPS=2

get_map() {
    case $1 in
        1) echo "" ;;
        2) echo "--map-by ppr:1:node" ;;
        4) echo "--map-by ppr:2:node" ;;
    esac
}

# Pre-flight: verify all graph files exist
for g in rmat-12 rmat-14 random-12 random-14; do
    if [ ! -f "$g" ]; then
        echo "ERROR: graph '$g' not found in $(pwd)"
        echo "Available graphs:"
        ls -1 rmat-* random-* 2>/dev/null
        exit 1
    fi
done
echo "All graph files present. Starting benchmark at $(date)"
echo ""

for graph in rmat-12 rmat-14 random-12 random-14; do
    for np in 1 2 4; do
        map=$(get_map $np)
        for ov in 0 1; do
            for r in $(seq 1 $REPS); do
                tag="${graph}_${np}p_ov${ov}_r${r}"
                echo ""
                echo "============ ${tag} ($(date +%H:%M:%S)) ============"
                BC_USE_OVERLAP=$ov mpiexec -n $np $map \
                    ./solution_mpi -in $graph -out /tmp/${tag}.res 2>&1
                rm -f /tmp/${tag}.res
            done
        done
    done
done

echo ""
echo "ALL DONE: $(date)"