#!/bin/bash
# T1.4 quench-point smoke test: one GPU, one rank, N = 4*30^3 = 108000 atoms
# at rho* = 0.95. Melt at T* = 2.0, quench to TQ, hold, optional seed RSEED.
# Runs as a job step inside an allocation (see smoke.slurm):
#   bash scripts/run_smoke.sh TQ RSEED [HOLD]
set -e
TQ=$1; RS=$2; HOLD=${3:-80000}
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
srun -n 1 --gres=gpu:1 --exact $ROOT/src/pmd_cuda smoke.in > smoke.log 2>&1
grep TIMING smoke.log
