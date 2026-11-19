# In situ analysis (Phase 2 T2.4, T2.5; Phase 3 T3.1)

The MD code (`src/pmd.c`, CUDA forces) runs on a SIM communicator, and a separate group of CPU
analysis (ANA) ranks labels every atom as liquid or crystal on the fly and tracks the largest
crystalline cluster. Only labels and summary rows are written, not trajectories.

## Build and run

```bash
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
make -C src                    # pmd, pmd_cuda, pmd_insitu (CUDA + -DINSITU), posthoc
make -C src pmd_insitu_cpu     # CPU forces + in situ (local tests; OMPFLAGS= without OpenMP)
srun -n <P_sim + P_ana> src/pmd_insitu <input>
```

Input keywords (in addition to the MD keywords in `docs/validation.md`):

| Keyword | Default | Meaning |
|---|---|---|
| `InsituRanks` | 0 | P_ana, # of ANA ranks (0 = analysis off, same binary) |
| `InsituInterval` | 100 | K, analysis every K steps |
| `InsituStart` | 0 | first step that may be analyzed |
| `InsituMovie` | 0 | full snapshot every M analysis steps (0 = off) |
| `InsituCheck` | 0 | 1 = full-precision dumps for the post hoc check |
| `InsituThr` | 0.308 | q̄6 threshold of the baseline labels |
| `InsituClusMin` | 20 | clusters with at least this many atoms are counted (`nbig`) |
| `InsituPrefix` | insitu | output file prefix |

The MPI size must be `vproc[0]*vproc[1]*vproc[2] + InsituRanks`.

## Design

```
MPI_COMM_WORLD (P_sim + P_ana ranks)
  MPI_Comm_split(color = rank < P_sim ? SIM : ANA)
    SIM: world ranks 0..P_sim-1, the MD code on simcomm (one GPU each)
    ANA: world ranks P_sim..P-1, CPU + OpenMP
  SIM rank s sends to ANA rank s*P_ana/P_sim
```

**SIM side, every K steps** (`insitu_send` in `pmd.c`, `insitu_sim_send` in `insitu.c`):

1. Copy the residents to separate arrays and run `halo_copy(2*r_q = 3.0, ...)`. This is
   `atom_copy` with the copy width as a parameter and the target arrays as arguments;
   `atom_copy()` is now `nb = halo_copy(RCUT, r, gid, NEMAX)`. The force cache `r[n:n+nb-1]`
   is not touched. A width of 2*r_q is what `steinhardt_compute` needs for exact q̄ at
   subdomain edges; the force cache (r_c = 2.5) is too thin.
2. `MPI_Wait` on the previous send (timed: this is the SIM stall), pack one `MPI_DOUBLE`
   message: a header (step, n_res, n_tot, subdomain origin in global coordinates, stall,
   halo and pack times, SIM rank), local positions of residents then halo atoms (the resident
   flag is the position in the block), and global IDs.
3. `MPI_Isend` to the ANA rank and keep integrating.

**ANA side** (`insitu_ana_run`):

1. For each assigned SIM rank in turn, `MPI_Probe` + `MPI_Get_count` + `MPI_Recv`. Receiving
   from specific sources keeps steps in order even when eager sends from fast SIM ranks arrive
   early.
2. `steinhardt_compute` on the block (OpenMP threads in passes 2 and 3), `steinhardt_features`,
   `mlp_classify_batch`, and the q̄6 > 0.308 baseline.
3. `MPI_Gatherv` of crystalline atoms (either label set: ID, global position, label bits) to ANA
   root.
4. Root sorts by ID and runs `cluster_find` (`src/cluster.c`: minimum image, bond if distance
   < r_q = 1.5, cell list, union-find) once per label set. It reports the largest and second
   largest cluster and the number of clusters with at least `InsituClusMin` atoms.
5. Root writes a summary row and appends to the label file; every M analysis steps it gathers
   all positions and writes a movie snapshot.

**Termination.** Both sides compute the same list of analysis steps from the input file
(`insitu_is_step`). Each SIM rank sends exactly one message per analysis step and waits for the
last one after the MD loop (`insitu_sim_finish`); each ANA rank receives exactly that many
messages from each of its SIM ranks and returns. There is at most one outstanding message per
SIM rank, so the end needs no extra handshake.

**GPU binding and placement.** The GPU is chosen by the rank index within the node on
*simcomm*, so ANA ranks never touch a GPU. On 2 nodes the job uses `srun --distribution=cyclic`:
world ranks 0..3 (SIM) land 2 per node, and ANA rank 4+s sits on the same node as SIM rank s.

## Output files

All binary files are little endian. `analysis/insitu_read.py` reads them.

| File | Content |
|---|---|
| `<prefix>_summary.dat` | one text row per analysis step (columns below) |
| `<prefix>_labels.bin` | per analysis step: `int32 step, int32 count, int32 id[count]` (ascending), `uint8 bits[count]`; crystalline atoms only |
| `<prefix>_movie_<step>.bin` | `int32 nglob, int32 step, double time, double box[3], float32 pos[nglob][3]` (ID order, wrapped), `uint8 bits[nglob]` |
| `<prefix>_full_<step>.bin` | check only: `int32 nglob, int32 step, double box[3], double pos[nglob][3]` in ID order (SIM rank 0) |
| `<prefix>_feat_<step>_a<k>.bin` | check only: in situ features of ANA rank k |

Label bits: 1 = MLP crystal, 2 = q̄6 > 0.308, 4 = in the largest MLP cluster, 8 = in the largest
threshold cluster.

Summary columns: `step time n_crystal_mlp n_crystal_thr nmax_mlp nmax_thr t_analysis t_stall
bytes_insitu | n2_mlp n2_thr nbig_mlp nbig_thr t_feat t_clus_io t_recvwait t_simhalo t_simpack
bytes_fulldump`.

- `t_analysis`: max over ANA ranks of (features + labels + gather), plus the root's cluster
  and I/O time. Time spent waiting for SIM data is excluded and reported as `t_recvwait`.
- `t_stall`: max over SIM ranks of the `MPI_Wait` before this step's send.
- `t_simhalo`, `t_simpack`: max over SIM ranks of the analysis halo exchange and packing time,
  the other SIM-side cost of the analysis.
- `bytes_insitu`: bytes written so far by the in situ analysis (summary, labels, movie).
- `bytes_fulldump`: bytes a full-precision dump of all atoms at every analysis step would have
  needed by now (positions as doubles, IDs implicit by order: 24 bytes/atom + 32-byte header,
  exactly the `_full_` format). A text XYZ at `%.17g` would be about 3x larger.

## T2.5 Equivalence check: in situ vs post hoc

With `InsituCheck 1`, SIM rank 0 also writes a full-precision binary dump of all atoms at every
analysis step (doubles, the exact bits the SIM ranks hold, plus the subdomain origin), and each
ANA rank dumps its in situ per-atom features. `src/posthoc` then, for every analysis step:

- reads the dump and runs `steinhardt_compute_periodic` on the whole periodic box (periodic
  images, no subdomains),
- labels with `mlp_classify_batch` and q̄6 > 0.308 and runs the same `cluster_find`,
- compares labels atom by atom (by global ID), largest-cluster membership, crystal counts and
  n_max with the in situ label file and summary, and the five features and p_crystal with the in
  situ feature dumps.

Runs: `scripts/insitu_check.slurm`, inputs `runs/insitu_check/check_{2x1,4x2}.in`. N = 32000 at
the production state point (ρ* = 0.95, T_q = 0.70, seed R = 4.5), 10k hold steps, K = 100,
analysis from step 6000 to 17000 (111 steps: end of the melt, the quench and the growing seed).
CUDA build `pmd_insitu`, debug node e23-02 (2x P100), 2 OpenMP threads per ANA rank.
Results in `runs/insitu_check/results/`.

| Run | Decomposition | Atom labels compared | MLP mismatches | Threshold mismatches | Largest-cluster membership mismatches | n_max / counts mismatching steps |
|---|---|---|---|---|---|---|
| 2 SIM + 1 ANA | vproc 1 1 2 | 3,552,000 | 0 | 0 | 0 | 0 of 111 |
| 4 SIM + 2 ANA | vproc 2 2 1 | 3,552,000 | 0 | 0 | 0 | 0 of 111 |

**Agreement is exact** for both label sets and for the cluster sizes (for example n_max at step
17000: MLP 1128, threshold 1427 in the 2+1 run; 1551 and 1906 in the 4+2 run; the two runs are
different trajectories because the decomposition changes the floating-point summation order).

The features themselves are not bit-identical, and should not be expected to be:

| Run | max \|Δq̄4\| | max \|Δq̄6\| | max \|Δq4\| | max \|Δq6\| | max \|Δp_crystal\| | coordination mismatches |
|---|---|---|---|---|---|---|
| 2 SIM + 1 ANA | 7.9e-16 | 1.4e-15 | 3.1e-15 | 4.7e-15 | 2.3e-14 | 0 |
| 4 SIM + 2 ANA | 8.7e-16 | 1.5e-15 | 3.0e-15 | 4.7e-15 | 2.8e-14 | 0 |

Two things differ between the paths at the last-digit level: (1) the pair vectors. In situ, a
halo atom carries its owner's local coordinate shifted by one subdomain length, while post hoc
uses global coordinates and periodic images shifted by the box length, so r_ij is rounded
differently; and (2) the order in which neighbors are summed, which follows each path's own cell
grid. Both are round-off at 1e-15 relative. A label could flip only if an atom's q̄6 were within
~1e-15 of 0.308, or its MLP logits within ~1e-14 of each other, or a pair distance within ~1e-15
of r_q = 1.5. With 7.1M atom labels compared, none occurred. The positions used for the cluster
search are bit-identical on both paths (local + origin is computed the same way), so given equal
labels the clusters must agree. The same check also passed locally (CPU build) for 1 SIM + 1 ANA,
2 + 1, 4 + 2 and 4 + 3 ranks (uneven mapping).

The dumps also confirm that every atom is analyzed exactly once per step (each ID appears in
exactly one ANA feature file).

The trajectory itself is not perturbed by the analysis: on a 4000-atom test, the `pmd_insitu_cpu`
energies match the original `pmd` to all printed digits over 5000 steps.

RESULTS_PLACEHOLDER
