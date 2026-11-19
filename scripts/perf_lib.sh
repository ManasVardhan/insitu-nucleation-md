# Helpers for the T3.2 performance jobs (sourced by scripts/perf_*.slurm).
#   source scripts/perf_lib.sh <experiment>
# Each run writes its input and log to runs/perf/out/<exp>_<jobid>/<tag>/
# and appends one row to runs/perf/results/<exp>_<jobid>.csv.
# SMOKE=1 in the environment makes the calling script run tiny versions.
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
PERF_EXP=$1
PERF_ROOT=$(pwd)
PERF_JOB=${SLURM_JOB_ID:-local}
[ "${SMOKE:-0}" = 1 ] && PERF_JOB=smoke_$PERF_JOB
PERF_OUT=$PERF_ROOT/runs/perf/out/${PERF_EXP}_$PERF_JOB
PERF_CSV=$PERF_ROOT/runs/perf/results/${PERF_EXP}_$PERF_JOB.csv
mkdir -p $PERF_OUT $PERF_ROOT/runs/perf/results
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
CPU_NAME=$(lscpu | sed -n 's/^Model name: *//p' | tr ' ' '_' | tr -s '_')
[ -f $PERF_CSV ] || echo "exp,config,bin,sim_ranks,ana_ranks,ana_threads,placement,K,nglob,steps,rep,step_ms,force_ms,comm_ms,wall_s,stall_total,stall_max,halo_total,pack_total,ana_steps,t_ana_mean,t_ana_max,bytes_insitu,bytes_fulldump,nodes,gpu,cpu,jobid" > $PERF_CSV
echo "perf $PERF_EXP job $PERF_JOB on $(hostname): gpu $GPU_NAME cpu $CPU_NAME"
nvidia-smi -L 2>/dev/null

# mk_md <file> "<vproc>" "<ucell>" <steps> [extra keyword lines...]
# Plain MD at rho* = 0.95 from an FCC lattice at T* = 2.0 (melting), NVE.
mk_md() {
  local f=$1 vp=$2 uc=$3 st=$4; shift 4
  { echo "vproc $vp"; echo "InitUcell $uc"; echo "Density 0.95"
    echo "InitTemp 2.0"; echo "DeltaT 0.005"; echo "StepLimit $st"
    echo "StepAvg $(( st < 100 ? st : 100 ))"
    for l in "$@"; do echo "$l"; done; } > $f
}

# mk_seeded <file> <K> <P_ana> <melt> <ramp> <hold> [vproc] [ucell]
# Seeded nucleation protocol of runs/production/prod.in (N = 1,048,576 by
# default), shortened; analysis every K steps, no movie, StepAvg 500.
mk_seeded() {
  local f=$1 K=$2 P=$3 sm=$4 sq=$5 sh=$6 vp=${7:-"2 2 1"} uc=${8:-"32 32 64"}
  { echo "vproc $vp"; echo "InitUcell $uc"; echo "Density 0.95"
    echo "InitTemp 2.0"; echo "DeltaT 0.005"; echo "StepAvg 500"
    echo "Thermostat 1"; echo "TauT 0.1"; echo "TMelt 2.0"; echo "StepMelt $sm"
    echo "TQuench 0.70"; echo "StepQuench $sq"; echo "StepHold $sh"
    echo "RSeed 4.5"; echo "StepRelease 500"
    echo "InsituRanks $P"; echo "InsituInterval $K"; echo "InsituStart 0"
    echo "InsituMovie 0"; echo "InsituCheck 0"; echo "InsituPrefix perf"; } > $f
}

# run <tag> <config> <bin> <rep> <input-file> <launch command...>
# Runs the launch command (e.g. srun -n 4 --cpu-bind=cores) on the binary
# with the input in its own directory and appends a CSV row.
run() {
  local tag=$1 cfg=$2 bin=$3 rep=$4 in=$5; shift 5
  local d=$PERF_OUT/$tag; mkdir -p $d; cp $in $d/run.in
  local L=$d/run.log T="" try
  # A launch occasionally hangs in MPI startup (seen once in MPI_Comm_split):
  # each attempt is killed after RUN_TMO s, and a failed run is retried once.
  for try in 1 2; do
    ( cd $d && timeout -s INT ${RUN_TMO:-900} "$@" $PERF_ROOT/src/$bin run.in > run.log 2>&1 )
    T=$(grep '^TIMING' $L)
    [ -n "$T" ] && break
    echo "FAILED $tag (attempt $try)"; tail -3 $L; cp $L $d/failed_$try.log
  done
  [ -z "$T" ] && return 1
  local nglob=$(echo $T | awk '{print $3}') np=$(echo $T | awk '{print $5}')
  local steps=$(echo $T | awk '{print $7}') wall=$(echo $T | awk '{print $9}')
  local sms=$(echo $T | awk '{print $11}') fms=$(echo $T | awk '{print $13}')
  local cms=$(echo $T | awk '{print $15}')
  local nana=$(awk '/^InsituRanks/{print $2}' $d/run.in); nana=${nana:-0}
  local K=$(awk '/^InsituInterval/{print $2}' $d/run.in)
  [ "$nana" = 0 ] && K=0
  local S=$(grep '^INSITU_SIM' $L) A=$(grep '^INSITU_ANA steps' $L)
  local st=$(echo $S | awk '{print $5}') sm=$(echo $S | awk '{print $7}')
  local ht=$(echo $S | awk '{print $11}') pt=$(echo $S | awk '{print $13}')
  local as=$(echo $A | awk '{print $3}') am=$(echo $A | awk '{print $7}')
  local ax=$(echo $A | awk '{print $9}') bi=$(echo $A | awk '{print $13}')
  local bf=$(echo $A | awk '{print $15}')
  echo "$PERF_EXP,$cfg,$bin,$np,$nana,${OMP_NUM_THREADS:-1},${PLACEMENT:-same},$K,$nglob,$steps,$rep,$sms,$fms,$cms,$wall,$st,$sm,$ht,$pt,$as,$am,$ax,$bi,$bf,${PERF_NODES:-$(hostname -s)},$GPU_NAME,$CPU_NAME,$PERF_JOB" >> $PERF_CSV
  echo "$tag $cfg N=$nglob P=$np ana=$nana K=$K step_ms=$sms force_ms=$fms comm_ms=$cms"
}
