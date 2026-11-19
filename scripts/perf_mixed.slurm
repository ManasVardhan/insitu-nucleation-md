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
# perf_speedup.slurm times 200 to 500 steps of NVE from the FCC lattice at
# InitTemp 2.0, which stays a hot solid (T ~ 1): atoms keep their lattice
# (memory) order. MELT=1 instead holds T* = 2.0 with the thermostat, so the
# box melts and the atoms mix as in the production run. WALL lines every
# 100 steps give the first and last 500-step windows. Layouts: 20-core CPU
# path, and 1 and 2 P100 with the old (host-order threads) and new
# (cell-order threads) kernel. SIZES = list of k:steps (N = 4000 k^3).
source scripts/perf_lib.sh mixed${MELT:+_melt}
export OMP_NUM_THREADS=1
ln -sf $OLD/src/pmd_cuda $PERF_ROOT/src/pmd_cuda_old
B="--cpu-bind=cores --distribution=block:block"
for kk in ${SIZES:-4:3000 6:2000}; do
  IFS=: read k S <<< "$kk"
  [ "${SMOKE:-0}" = 1 ] && S=200
  a=$((2*k)); b=$((5*k)); g=$((10*k)); N=$((4000*k*k*k))
  I=$PERF_OUT/in_$k; mkdir -p $I
  X=(); [ "${MELT:-0}" = 1 ] && X=("Thermostat 1" "TauT 0.1" "TMelt 2.0" "StepMelt $S")
  mk_md $I/gpu1.in "1 1 1" "$g $g $g" $S "${X[@]}"
  mk_md $I/gpu2.in "1 1 2" "$g $g $b" $S "${X[@]}"
  mk_md $I/cpu20.in "5 2 2" "$a $b $b" $S "${X[@]}"
  for c in "cpu20:pmd:20" "gpu1_old:pmd_cuda_old:1" "gpu1_new:pmd_cuda:1" \
           "gpu2_old:pmd_cuda_old:2" "gpu2_new:pmd_cuda:2"; do
    IFS=: read tag bin np <<< "$c"
    run ${tag}_N$N $tag $bin 1 $I/${tag%%_*}.in srun -n $np $B
    grep '^WALL' $PERF_OUT/${tag}_N$N/run.log > $PERF_ROOT/runs/perf/results/mixed${MELT:+melt}_${PERF_JOB}_${tag}_N$N.wall
  done
done
echo done $(date)
