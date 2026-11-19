#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-speedup-%j.out
# T3.2 measurement 1: CPU vs CUDA force path, step time vs N (repo root):
#   sbatch scripts/perf_speedup.slurm          (SMOKE=1 sbatch ... for a short test)
# One node (debug e23-02: 2x P100, 2x Xeon E5-2640 v4, 10 cores each).
# Global box 10k x 10k x 10k FCC cells (N = 4000 k^3), the same for every
# decomposition: cpu1 = 1 core, cpu10 = one socket (10 ranks), cpu20 =
# the whole node (20 ranks), gpu1 = 1 rank + 1 P100, gpu2 = 2 ranks + 2 P100.
source scripts/perf_lib.sh speedup
export OMP_NUM_THREADS=1
KS=${KS:-"2 3 4 5 6"}; REPS=${REPS:-3}
SG=500; SC=200
if [ "${SMOKE:-0}" = 1 ]; then KS="2 6"; REPS=1; SG=20; SC=10; fi
for rep in $(seq 1 $REPS); do
  for k in $KS; do
    a=$((2*k)); b=$((5*k)); g=$((10*k)); N=$((4000*k*k*k))
    case $k in 2|3) S1=100;; 4) S1=50;; *) S1=30;; esac
    [ "${SMOKE:-0}" = 1 ] && S1=5
    I=$PERF_OUT/in_$k; mkdir -p $I
    mk_md $I/gpu1.in "1 1 1" "$g $g $g" $SG
    mk_md $I/gpu2.in "1 1 2" "$g $g $b" $SG
    mk_md $I/cpu20.in "5 2 2" "$a $b $b" $SC
    mk_md $I/cpu10.in "5 2 1" "$a $b $g" $SC
    mk_md $I/cpu1.in "1 1 1" "$g $g $g" $S1
    run gpu1_N${N}_r$rep gpu1 pmd_cuda $rep $I/gpu1.in srun -n 1 --cpu-bind=cores
    run gpu2_N${N}_r$rep gpu2 pmd_cuda $rep $I/gpu2.in srun -n 2 --cpu-bind=cores
    run cpu20_N${N}_r$rep cpu20 pmd $rep $I/cpu20.in srun -n 20 --cpu-bind=cores --distribution=block:block
    run cpu10_N${N}_r$rep cpu10 pmd $rep $I/cpu10.in srun -n 10 --cpu-bind=cores --distribution=block:block
    [ $rep -le 2 ] && run cpu1_N${N}_r$rep cpu1 pmd $rep $I/cpu1.in srun -n 1 --cpu-bind=cores
  done
done
echo done $(date)
