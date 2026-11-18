# Plan

Dates: today 2026-11-18. Instructor meeting by Nov 20. Presentation Nov 30 or Dec 1/2. Repo due Dec 11.
Each task maps to one GitHub Issue. See `design.md` for the technical spec.

## Phase 0: Setup (Nov 18-20)
- **T0.1** Repo, README, design + plan docs, Issues board.
- **T0.2** Abstract (Problem / Finding / So what) for the instructor meeting.
- **T0.3** Bring course `pmd.c` into `src/`, build on CARC, run baseline; record CARC GPU partitions and limits.

## Phase 1: GPU MD (Nov 20-23)
- **T1.1** Velocity-rescaling thermostat + melt/quench protocol driven by input file.
- **T1.2** CUDA cell-list force kernel behind `-DUSE_CUDA`; CPU path kept.
- **T1.3** Validation: GPU vs CPU forces, NVE energy drift, liquid g(r), thermostat settling.
- **T1.4** Smoke test to pick quench point (ρ*, T*) and seed size.

## Phase 2: In situ analysis (Nov 23-26)
- **T2.1** Steinhardt q4/q6 + Lechner-Dellago averages in C (`steinhardt.c`), unit-tested on perfect FCC.
- **T2.2** Training data generation (FCC vs liquid), PyTorch MLP, weight export to C header.
- **T2.3** C forward pass (`mlp.c`), checked against PyTorch outputs.
- **T2.4** `MPI_Comm_split` SIM/ANA, non-blocking position+halo transfer, labeling, largest-cluster tracking.
- **T2.5** In situ vs post hoc equivalence check.

## Phase 3: Results (Nov 26-29)
- **T3.1** Production seeded run on CARC GPUs.
- **T3.2** GPU speedup, strong and weak scaling, in situ overhead, I/O savings runs.
- **T3.3** Plots (`analysis/`) and OVITO movie (`viz/`).
- **T3.4** Presentation slides.

## Phase 4: Polish (Dec 1-11)
- **T4.1** README as report: story, method, results, reproduction steps.
- **T4.2** Optional: unseeded homogeneous nucleation run; device-resident positions.
- **T4.3** Final submission.

## Execution
Subagent-driven: each task implemented by an Opus subagent, reviewed before merge. CARC jobs are
submitted only after review. Smoke test on small N before any large run.
