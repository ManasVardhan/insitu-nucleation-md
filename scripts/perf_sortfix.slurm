#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -c 2
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-sortfix-%j.out
# Force kernel with threads in cell-sorted order (repo root):
#   OLD=<dir with the previous src/pmd_cuda> sbatch scripts/perf_sortfix.slurm
# 1. Bit-identity: old vs new pmd_cuda on a 2-rank seeded run (energies
#    printed to 8 decimals every 100 steps must match exactly).
# 2. The perf_drift.slurm protocol (15000 steps, analysis off and on) with
#    the new kernel, to compare with the drift job of the old kernel.
source scripts/perf_lib.sh sortfix
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
I=$PERF_OUT/in; mkdir -p $I
mk_seeded $I/ident.in 100 0 1000 500 1500 "1 1 2" "16 16 8"
sed -i 's/^StepAvg 500/StepAvg 100/' $I/ident.in
for b in old new; do
  B=$PERF_ROOT/src/pmd_cuda; [ $b = old ] && B=$OLD/src/pmd_cuda
  mkdir -p $PERF_OUT/ident_$b && cp $I/ident.in $PERF_OUT/ident_$b/run.in
  ( cd $PERF_OUT/ident_$b && srun -n 2 -c 2 --cpu-bind=cores $B run.in > run.log 2>&1 )
done
grep -E '^ *[0-9]+ ' $PERF_OUT/ident_old/run.log > $PERF_OUT/e_old.txt
grep -E '^ *[0-9]+ ' $PERF_OUT/ident_new/run.log > $PERF_OUT/e_new.txt
echo "energy lines: $(wc -l < $PERF_OUT/e_old.txt) old, $(wc -l < $PERF_OUT/e_new.txt) new"
if cmp -s $PERF_OUT/e_old.txt $PERF_OUT/e_new.txt; then echo "IDENTICAL energies"; else echo "DIFFERENT energies"; diff $PERF_OUT/e_old.txt $PERF_OUT/e_new.txt | head; fi
grep TIMING $PERF_OUT/ident_*/run.log
SM=5000; SQ=2000; SH=8000
mk_seeded $I/off.in 100 0 $SM $SQ $SH
mk_seeded $I/on.in 100 4 $SM $SQ $SH
for c in off:0 on:4; do
  IFS=: read tag P <<< "$c"
  run $tag $tag pmd_insitu 1 $I/$tag.in srun -n $((4+P)) -c 2 --cpu-bind=cores
  grep '^WALL' $PERF_OUT/$tag/run.log > $PERF_ROOT/runs/perf/results/sortfix_${PERF_JOB}_$tag.wall
done
echo done $(date)
