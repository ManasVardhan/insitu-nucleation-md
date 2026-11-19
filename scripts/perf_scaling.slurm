#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-scaling-%j.out
# T3.2 measurements 2-3 on one node (repo root): sbatch scripts/perf_scaling.slurm
# GPU strong scaling at N = 1,048,576 and 256,000 (1 and 2 P100; 4 ranks on
# 2 P100 is the oversubscribed production layout), GPU weak scaling at
# 262,144 atoms per rank, and the same on the CPU path (1 to 20 cores):
# strong at N = 256,000, weak at 32,000 atoms per rank.
# The multi-node GPU version is scripts/perf_scaling_multi.slurm.
source scripts/perf_lib.sh scaling
export OMP_NUM_THREADS=1
REPS=${REPS:-3}; SG=300; SC=100; SW=200
if [ "${SMOKE:-0}" = 1 ]; then REPS=1; SG=10; SC=5; SW=5; fi
B="--cpu-bind=cores --distribution=block:block"
I=$PERF_OUT/in; mkdir -p $I
mk_md $I/gs1M_1.in "1 1 1" "64 64 64" $SG
mk_md $I/gs1M_2.in "1 1 2" "64 64 32" $SG
mk_md $I/gs1M_4.in "2 2 1" "32 32 64" $SG
mk_md $I/gs256k_1.in "1 1 1" "40 40 40" $SG
mk_md $I/gs256k_2.in "1 1 2" "40 40 20" $SG
mk_md $I/gw_1.in "1 1 1" "32 32 64" $SG
mk_md $I/gw_2.in "2 1 1" "32 32 64" $SG
for p in "1:1 1 1:40 40 40" "2:1 1 2:40 40 20" "4:2 2 1:20 20 40" "8:2 2 2:20 20 20" \
         "16:4 2 2:10 20 20" "20:5 2 2:8 20 20"; do
  IFS=: read P vp uc <<< "$p"
  mk_md $I/cs_$P.in "$vp" "$uc" $SC
  mk_md $I/cw_$P.in "$vp" "20 20 20" $SW
done
for rep in $(seq 1 $REPS); do
  for P in 1 2 4; do
    run gs1M_${P}_r$rep gpu_strong_1M pmd_cuda $rep $I/gs1M_$P.in srun -n $P $B
  done
  for P in 1 2; do
    run gs256k_${P}_r$rep gpu_strong_256k pmd_cuda $rep $I/gs256k_$P.in srun -n $P $B
    run gw_${P}_r$rep gpu_weak_262k pmd_cuda $rep $I/gw_$P.in srun -n $P $B
  done
  for P in 1 2 4 8 16 20; do
    [ $rep -le 2 ] && run cs_${P}_r$rep cpu_strong_256k pmd $rep $I/cs_$P.in srun -n $P $B
    run cw_${P}_r$rep cpu_weak_32k pmd $rep $I/cw_$P.in srun -n $P $B
  done
done
echo done $(date)
