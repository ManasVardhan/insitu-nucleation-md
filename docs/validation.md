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

## T1.4 Quench point and seed size

Smoke tests with `scripts/smoke.slurm` / `scripts/run_smoke.sh`: N = 108000 (30^3 FCC cells),
ρ* = 0.95, one rank on one P100 per run (7.3 ms/step). Protocol: melt at T* = 2.0 for 5000
steps, linear ramp to T_q over 2000 steps, hold. The hold starts at step 7000. Seeds are
frozen until step 8000 (StepRelease = 1000), then released. Crystallinity:

- **PE proxy:** change of PE per atom relative to the start of the hold (mean over hold
  steps 1000 to 3000), at constant T. Crystallization releases latent heat, so PE drops;
  full crystallization is about -0.4.
- **Cross-check:** `analysis/solid_frac.py` (ten Wolde q6 criterion, r_q = 1.5): fraction of
  solid-like atoms and the largest solid cluster. A perfect FCC frame gives 1.0, and the
  equilibrated liquid gives 1e-4.

ΔPE per atom vs hold steps (from `runs/smoke/results/summary.txt`):

| T_q | R_seed (atoms) | 10k | 20k | 30k | 40k | 50k | 60k | 80k | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 0.55 | none | -0.007 | -0.007 | -0.020 | -0.062 | -0.151 | -0.250 | -0.302 | nucleates on its own |
| 0.55 | 4.0 (249) | -0.015 | -0.057 | -0.172 | -0.335 | -0.411 | -0.422 | -0.431 | grows (fast) |
| **0.70** | **none** | +0.004 | +0.004 | -0.001 | +0.002 | +0.002 | +0.001 | +0.006 | **stays liquid** |
| 0.70 | 2.5 (55) | +0.001 | -0.002 | +0.003 | -0.001 | +0.004 | | | melts |
| 0.70 | 3.0 (87) | -0.001 | -0.002 | +0.004 | +0.004 | +0.003 | | | melts |
| 0.70 | 4.0 (249) | -0.002 | -0.021 | -0.070 | -0.165 | -0.283 | -0.363 | -0.405 | grows |
| 0.70 | 4.0 (249), 2nd velocity seed | -0.005 | -0.028 | -0.089 | -0.193 | -0.310 | -0.364 | | grows |
| **0.70** | **4.5 (369)** | -0.003 | -0.038 | -0.126 | -0.252 | -0.349 | -0.395 | | **grows** |
| 0.85 | none | +0.004 | +0.004 | +0.004 | -0.001 | +0.001 | -0.001 | -0.003 | liquid |
| 0.85 | 4.0 (249) | +0.001 | -0.001 | +0.004 | -0.002 | -0.002 | +0.003 | -0.003 | seed melts |

Largest solid cluster (q6 criterion) at absolute steps 10k, 20k, 30k, 40k, 50k, 60k, 80k:

| Run | 10k | 20k | 30k | 40k | 50k | 60k | 80k |
|---|---|---|---|---|---|---|---|
| T 0.55, no seed | 33 | 152 | 945 | 3779 | | 44882 | 71635 |
| T 0.70, no seed | 8 | 6 | 8 | 7 | 12 | 12 | 9 |
| T 0.70, R 3.0 | 155 | 7 | 6 | 10 | 8 | | |
| T 0.70, R 4.0 | 676 | 2243 | 6753 | 18384 | 38944 | 62942 | 80770 |
| T 0.70, R 4.0 (2nd seed) | 646 | 2662 | 8245 | 23541 | 45952 | 67216 | |
| T 0.70, R 4.5 | 777 | 3390 | 12656 | 32140 | 57760 | 73450 | |
| T 0.85, R 4.0 | | | | 3 | | | 5 |

**Chosen parameters: ρ* = 0.95, T_q = 0.70, R_seed = 4.5 (369 atoms), StepRelease = 1000.**

- At the starting guess T_q = 0.55 the unseeded liquid nucleates homogeneously within about
  15k to 25k hold steps (cluster of 152 at step 20k, 945 at 30k), which would compete with
  the seed. It is too deep.
- At T_q = 0.70 the unseeded liquid shows no drift in PE and no solid cluster larger than 12
  atoms over 80k hold steps, while a seed of R = 4.0 or 4.5 grows steadily: the crystal
  passes half the box between 40k and 60k hold steps. This is the window the in situ
  analysis needs.
- The critical seed at T_q = 0.70 lies between 87 atoms (R = 3.0, cluster of 155 at step 10k, gone by step 20k)
  and 249 atoms (R = 4.0, grows in 2 of 2 realizations). R = 4.5 adds margin against
  a stochastic melt of a near-critical seed, and grows a little faster.
- At T_q = 0.85 even the R = 4.0 seed melts, which brackets the useful range from above.
- The planned T_q = 1.00 runs were cancelled to free the debug node, since 0.85 already
  shows seed melting.

Ready-to-use inputs: `runs/nucleation/seeded.in` (2 GPUs, vproc 1 1 2, 108000 atoms, 80k hold,
XYZ every 5000 steps) and `runs/nucleation/unseeded.in` (control). A short 2-GPU test of
`seeded.in` (P100) confirmed 369 seed atoms straddling the rank boundary stay frozen
until release and move after it, and GPU vs CPU forces agree to 7e-14. Step time is
4.0 ms, so the full 87k-step run takes about 6 minutes on 2 P100s.

Caveats: the smoke tests ran on one rank, the production input on two (same physics, as
checked above). The seed's surroundings partly order while it is frozen (cluster of
about 650 atoms 2000 steps after release for R = 4.0), so the effective initial nucleus
is larger than the frozen seed. Each case was run once, except R = 4.0 at T_q = 0.70,
which was run twice.
