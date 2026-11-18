#!/bin/bash
# T1.4 quench-point smoke test: one GPU, one rank, N = 4*30^3 = 108000 atoms
# at rho* = 0.95. Melt at T* = 2.0, quench to TQ, hold, optional seed RSEED.
# Usage (repo root, login node): bash scripts/run_smoke.sh TQ RSEED [HOLD] [PART]
set -e
TQ=$1; RS=$2; HOLD=${3:-80000}; PART=${4:-gpu}
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
ROOT=$(pwd)
D=runs/smoke/T${TQ}_R${RS}
mkdir -p $D && cd $D
cat > smoke.in <<EOT
vproc       1 1 1
InitUcell   30 30 30
Density     0.95
InitTemp    2.0
DeltaT      0.005
StepAvg     500
Thermostat  1
TauT        0.1
TMelt       2.0
StepMelt    5000
TQuench     $TQ
StepQuench  2000
StepHold    $HOLD
RSeed       $RS
StepRelease 1000
XYZInterval 10000
XYZPrefix   smoke
EOT
srun -p $PART -A anakano_429 -n 1 --gres=gpu:1 -t 00:30:00 \
  $ROOT/src/pmd_cuda smoke.in > smoke.log 2>&1
grep TIMING smoke.log
