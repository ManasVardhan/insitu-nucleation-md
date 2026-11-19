#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-mixed-%j.out
# CPU vs GPU once the atoms are mixed (repo root):
#   OLD=<dir with the host-order-thread src/pmd_cuda> sbatch scripts/perf_mixed.slurm
# perf_speedup.slurm times 200 to 500 steps from the FCC lattice, while
# atoms are still close to lattice (memory) order. Here each layout runs
# 3000 (N = 256,000) or 2000 (N = 864,000) steps at T* = 2.0 with WALL
# lines every 100 steps, so the first and last windows give the step time
# in ordered and mixed memory order: 20-core CPU path, and 1 and 2 P100
# with the old (host-order threads) and new (cell-order threads) kernel.
source scripts/perf_lib.sh mixed
export OMP_NUM_THREADS=1
ln -sf $OLD/src/pmd_cuda $PERF_ROOT/src/pmd_cuda_old
B="--cpu-bind=cores --distribution=block:block"
for kk in 4:3000 6:2000; do
  IFS=: read k S <<< "$kk"
  [ "${SMOKE:-0}" = 1 ] && S=200
  a=$((2*k)); b=$((5*k)); g=$((10*k)); N=$((4000*k*k*k))
  I=$PERF_OUT/in_$k; mkdir -p $I
  mk_md $I/gpu1.in "1 1 1" "$g $g $g" $S
  mk_md $I/gpu2.in "1 1 2" "$g $g $b" $S
  mk_md $I/cpu20.in "5 2 2" "$a $b $b" $S
  for c in "cpu20:pmd:20" "gpu1_old:pmd_cuda_old:1" "gpu1_new:pmd_cuda:1" \
           "gpu2_old:pmd_cuda_old:2" "gpu2_new:pmd_cuda:2"; do
    IFS=: read tag bin np <<< "$c"
    run ${tag}_N$N $tag $bin 1 $I/${tag%%_*}.in srun -n $np $B
    grep '^WALL' $PERF_OUT/${tag}_N$N/run.log > $PERF_ROOT/runs/perf/results/mixed_${PERF_JOB}_${tag}_N$N.wall
  done
done
echo done $(date)
