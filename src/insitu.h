/*----------------------------------------------------------------------
insitu.h: in situ phase analysis for pmd.c (design.md section 3.2).

MPI_COMM_WORLD is split into nsim SIM ranks (world ranks 0..nsim-1, the
MD code) and nana ANA ranks (world ranks nsim..nsim+nana-1, CPU only).
SIM rank s sends to ANA rank s*nana/nsim.  Every `interval` steps each SIM
rank sends its residents plus an analysis-only halo of width 2*rq (so the
Lechner-Dellago averages are exact at subdomain edges) with MPI_Isend and
keeps integrating.  ANA ranks compute Steinhardt features, MLP and
qbar6-threshold labels, gather the crystalline atoms to ANA root, and find
the largest crystalline cluster for both label sets.

SIM -> ANA message (MPI_DOUBLE, tag INSITU_TAG, on MPI_COMM_WORLD):
  hdr[INSITU_HDR]: step, nres, ntot, origin x y z (global coordinates of
    the subdomain corner), t_wait (stall before this send), t_halo,
    t_pack, sim rank
  pos[3*ntot]: local coordinates; atoms 0..nres-1 are residents (the
    resident flag is the position in the block), nres..ntot-1 are halo
  gid[ntot]: global IDs (as doubles, exact)

Output files (prefix = InsituPrefix), all little-endian binary except the
summary:
  <prefix>_summary.dat   one text row per analysis step (see header line)
  <prefix>_labels.bin    per analysis step: int32 step, int32 count,
                         int32 id[count] (ascending), uint8 bits[count]
                         for crystalline atoms only (either label set)
  <prefix>_movie_<step>.bin   every `movie` analysis steps, all atoms in ID
                         order: int32 nglob, int32 step, double time,
                         double box[3], float32 pos[nglob][3] (global,
                         wrapped), uint8 bits[nglob]
  bits: 1 = MLP crystal, 2 = qbar6 > thr, 4 = in largest MLP cluster,
        8 = in largest threshold cluster
Equivalence check (check = 1), not counted as in situ output:
  <prefix>_full_<step>.bin   written by SIM rank 0: int32 nglob, int32
                         step, double box[3], double pos[nglob][3] in ID
                         order (global, unwrapped: local + origin)
  <prefix>_feat_<step>_a<k>.bin  written by ANA rank k: int32 count,
                         int32 id[c], double feat[c][5], double pcr[c],
                         uint8 bits[c] (bits 1 and 2 only)
----------------------------------------------------------------------*/
#ifndef INSITU_H
#define INSITU_H

#include "mpi.h"

#define INSITU_TAG 700
#define INSITU_HDR 10
#define INSITU_FULL_HDR 32 /* bytes of the _full_ header */

typedef struct {
  int nsim, nana;      /* # of SIM & ANA ranks */
  int interval;        /* analysis every `interval` steps (K) */
  int start, last;     /* analysis steps s: start <= s <= last, s > 0 */
  int movie;           /* movie snapshot every `movie` analysis steps */
  int check;           /* 1 = write equivalence-check dumps */
  int clusmin;         /* clusters >= clusmin atoms are counted */
  int nglob;           /* total # of atoms */
  double rq;           /* Steinhardt neighbor cutoff */
  double thr;          /* qbar6 threshold baseline */
  double dt;           /* time step */
  double box[3];       /* global box lengths */
  char prefix[256];
} insitu_cfg;

/* Is `step` an analysis step? */
int insitu_is_step(const insitu_cfg *c, int step);
/* # of analysis steps */
int insitu_nsteps(const insitu_cfg *c);

/* SIM side.  x: positions of ntot atoms (residents first), g: global IDs,
   org: subdomain origin, t_halo: seconds spent in the analysis halo
   exchange.  Waits for the previous send (timed), packs & MPI_Isend. */
void insitu_sim_send(const insitu_cfg *c, int step, int nres, int ntot,
                     const double *x, const int *g, const double org[3],
                     double t_halo);
/* Waits for the last send and prints SIM-side totals (sim comm rank 0) */
void insitu_sim_finish(const insitu_cfg *c, MPI_Comm sim);
/* Equivalence check: full-precision dump of all residents (collective) */
void insitu_sim_dump(const insitu_cfg *c, MPI_Comm sim, int step, int n,
                     const double *x, const int *g, const double org[3]);

/* ANA side: runs all analysis steps, then returns */
void insitu_ana_run(const insitu_cfg *c, MPI_Comm ana);

#endif
