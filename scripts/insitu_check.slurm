#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH -t 00:30:00
#SBATCH -o runs/insitu_check/slurm-%j.out
# T2.5 in situ vs post hoc equivalence check (repo root):
#   sbatch scripts/insitu_check.slurm
# Runs 2 SIM + 1 ANA and 4 SIM + 2 ANA, then src/posthoc on each.
# Outputs in runs/insitu_check/out_<cfg>/.
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
export OMP_NUM_THREADS=2
ROOT=$(pwd)
nvidia-smi -L
for cfg in 2x1:3 4x2:6; do
  name=${cfg%%:*}; np=${cfg##*:}
  D=runs/insitu_check/out_$name
  mkdir -p $D && cd $D
  srun -n $np $ROOT/src/pmd_insitu ../check_$name.in > run.log 2>&1
  $ROOT/src/posthoc check > posthoc.log
  echo "== $name"; grep -E "TIMING|INSITU_SIM|INSITU_ANA steps" run.log; tail -3 posthoc.log
  cd $ROOT
done
