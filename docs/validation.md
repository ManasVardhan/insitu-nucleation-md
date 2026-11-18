# Validation (Phase 1)

All runs on CARC Discovery. Toolchain: `gcc/13.3.0`, `openmpi/5.0.5`, `cuda/12.6.3`.
Reduced LJ units, force-shifted LJ with r_c = 2.5, Δt = 0.005.

## Build and run

```bash
# on the CARC login node
git clone https://github.com/ManasVardhan/insitu-nucleation-md.git
cd insitu-nucleation-md
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
make -C src            # builds src/pmd (CPU) and src/pmd_cuda (-DUSE_CUDA)

# validation (force check, NVE, g(r), thermostat), then CPU vs GPU timing
salloc -p debug -A anakano_429 -N 1 -n 4 --gres=gpu:2 -t 00:40:00 bash scripts/run_validation.sh
salloc -p debug -A anakano_429 -N 1 -n 2 --gres=gpu:2 -t 00:30:00 bash scripts/run_timing.sh

# analysis (numpy; solid_frac.py also needs scipy)
python analysis/energy_check.py runs/validation/out/nve_gpu.log runs/validation/out/nve_cpu.log
python analysis/gr.py runs/validation/out/nve_0000{5000,5500,6000,6500,7000}.xyz
```

A single run is `srun -n <ranks> src/pmd_cuda <input>`; the input is a keyword file
(see `src/pmd.in`, `runs/validation/*.in`). Logs and data from the runs below are in
`runs/validation/results/`.

## T0.3 baseline

Unmodified course `pmd.c`, 2 ranks (vproc 1 1 2), 16^3 unit cells per rank (N = 32768),
ρ* = 0.8, 200 steps, debug node e23-02 (Xeon E5-2640 v4): **6.69 s, 33.5 ms/step**.

After adding global IDs and the keyword input, trajectories are bit-identical to the
original code (same energies to all printed digits, 2 ranks, 400 steps), and a
`-DCHECK_IDS` build confirms at every report step that each global ID is owned by exactly
one rank and that every cached copy sits at a periodic image of its resident
(0 errors, 4 ranks, T* = 2).

## Summary

| Check | Criterion | Result | Pass |
|---|---|---|---|
| GPU vs CPU forces | max relative error < 1e-10 | 8.0e-14 max per-atom \|ΔF\|/\|F\| (2 and 4 ranks) | yes |
| NVE drift (thermostat off) | < 1e-4 per atom over 10k steps | GPU 1.8e-5 (end - start), fit 6.5e-7; CPU 1.7e-5, fit 6.8e-7 | yes |
| Liquid g(r) | peaks match standard LJ liquid | 1st peak 1.085 (g = 3.04), 1st min 1.535, 2nd peak 2.045 | yes |
| Thermostat | settles within 1% of target | mean T 0.72198 vs 0.722 (-3.4e-5); within 1% after ~140 steps | yes |

## GPU vs CPU forces

The CUDA build recomputes the forces of the current configuration with the CPU loop every
`ForceCheck` steps and compares (`force_check()` in `pmd.c`). Inputs:
`runs/validation/force_check.in` (2 ranks, 6912 atoms, vproc 1 1 2) and `force_check4.in`
(4 ranks on 2 GPUs, 12288 atoms, vproc 2 2 1), through melt at T* = 2, quench to 0.55 and
a seeded hold at ρ* = 0.95. Tesla P100.

| Run | max \|ΔF\|/\|F\| per atom | max \|ΔF\| / max \|F\| | max per-atom PE rel. err. | total PE rel. err. |
|---|---|---|---|---|
| 2 ranks, 6 checks | ≤ 3.3e-14 | ≤ 1.9e-15 | ≤ 1.8e-12 | ≤ 1.2e-14 |
| 4 ranks, 6 checks | ≤ 8.0e-14 | ≤ 2.1e-15 | ≤ 5.4e-12 | ≤ 6.8e-15 |

Frozen seed atoms are excluded from the per-atom relative force error: inside a perfect
lattice their net force is ~1e-13 by symmetry, so the ratio is noise (3e-3), and their
forces are never used. The largest per-atom PE relative errors come from atoms whose PE
is close to zero at T* = 2. The total PE agrees to round-off, which confirms the
half-pair PE split on the GPU matches the CPU's i<j accounting.

## NVE energy conservation

`runs/validation/nve.in`: N = 16384, ρ* = 0.8442; melt at T* = 2 (2000 steps), ramp to
0.722 (1000), NVT hold (4000), then **10000 NVE steps** with the thermostat off.
Energies per atom, sampled every 10 steps.

| Build | E0 | E_end - E0 | max \|E - E0\| | linear-fit drift over 10k | std | mean T |
|---|---|---|---|---|---|---|
| pmd_cuda | -3.46885084 | -1.79e-5 | 5.92e-5 | -6.5e-7 | 1.40e-5 | 0.7224 |
| pmd (CPU) | -3.46992683 | -1.69e-5 | 6.07e-5 | +6.8e-7 | 1.53e-5 | 0.7218 |

Both are well below 1e-4 per atom. The GPU and CPU runs start from the same configuration
and agree to all 8 printed digits of E for the first 690 steps, then separate because of
round-off and chaos, which is why their E0 differ after 7000 steps.

## Liquid g(r)

Same run, 5 frames from the NVT hold (steps 5000 to 7000, T* = 0.722, ρ* = 0.8442).
`analysis/gr.py`, bin 0.01, data in `runs/validation/results/gr_liquid.dat`.

| Feature | This code | Standard LJ liquid near the triple point |
|---|---|---|
| 1st peak | r = 1.085, g = 3.04 | r ≈ 1.09 to 1.10, g ≈ 3.0 |
| 1st minimum | r = 1.535, g = 0.57 | r ≈ 1.5 to 1.6 |
| 2nd peak | r = 2.045, g = 1.28 | r ≈ 2.0 to 2.1 |

Reference values are the textbook LJ liquid at ρ* ≈ 0.844, T* ≈ 0.72 to 0.73 (Verlet 1968;
Frenkel and Smit, Case Study 4). These come from the full LJ potential; force shifting at
r_c = 2.5 changes the structure only slightly. The first minimum, 1.5, is the r_q we will
use for the Steinhardt neighbor shell.

## Thermostat

Berendsen, τ_T = 0.1. In the NVT hold of the run above (target 0.722), the 100-step running
mean of T is within 1% of the target 140 steps (0.7 time units) after the end of the ramp
(CPU: 130). Over the second half of the hold, mean T = 0.72198 (relative deviation -3.4e-5),
std 0.0025, and 98% of the instantaneous samples are within 1%. In the ρ* = 0.95 quench
runs, T settles at 0.551 for target 0.55 (`force_check.log`).

## Preliminary CPU vs GPU step time

2 ranks (vproc 1 1 2) on one node, ρ* = 0.95, NVE, 200 steps. Node e23-02:
Xeon E5-2640 v4 (one core per rank) and Tesla P100 (one per rank). `scripts/run_timing.sh`,
raw output in `runs/validation/results/timing.txt`.

| N | CPU step (ms) | GPU step (ms) | GPU force (ms) | Speedup (step) |
|---|---|---|---|---|
| 2048 | 2.06 | 0.71 | 0.59 | 2.9x |
| 6912 | 7.07 | 0.97 | 0.75 | 7.3x |
| 16384 | 13.77 | 0.98 | 0.63 | 14.0x |
| 55296 | 46.76 | 1.70 | 0.83 | 27.4x |
| 131072 | 105.87 | 3.43 | 1.63 | 30.9x |

The GPU force time includes the H→D/D→H copies and the cell sort. At larger N the
remaining host work (atom_copy/atom_move packing, integration) is about half of the GPU
step time, so it is the next bottleneck. A100/A40 numbers come in Phase 3.
