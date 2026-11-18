/*
 * steinhardt.h: per-atom Steinhardt bond-order parameters (l = 4, 6) and
 * Lechner-Dellago neighbor-averaged versions, plus coordination number.
 *
 * Plain C99, no external dependencies besides libm.
 *
 * Definitions (neighbors j of atom i are all atoms with |r_ij| < r_q):
 *   q_lm(i)    = 1/N_b(i) * sum_j Y_lm(r_ij)
 *   q_l(i)     = sqrt( 4 pi/(2l+1) * sum_m |q_lm(i)|^2 )
 *   qbar_lm(i) = 1/(N_b(i)+1) * sum_{k in {i} U nb(i)} q_lm(k)
 *   qbar_l(i)  = sqrt( 4 pi/(2l+1) * sum_m |qbar_lm(i)|^2 )
 *   coord(i)   = N_b(i)
 * An atom with no neighbors gets q_lm = 0 (so q_l = 0).
 *
 * Coordinates are LOCAL and NON-PERIODIC: no minimum-image convention is
 * applied anywhere in steinhardt_compute(). The input set is
 *   residents: atoms whose features are wanted, and
 *   halo atoms: atoms that only supply neighbors.
 *
 * HALO REQUIREMENT: for the results of every resident to be exact, the
 * input must contain every atom within distance 2*r_q of any resident
 * (q_lm of a resident's neighbor needs that neighbor's full shell). In
 * practice: a halo layer of thickness >= 2*r_q around the resident region.
 * A thinner halo gives silently wrong q/qbar for residents near the edge.
 *
 * Periodic systems: build explicit image atoms with
 * steinhardt_periodic_images() (halo = 2*r_q) and call the same
 * steinhardt_compute(), or use the wrapper steinhardt_compute_periodic().
 * Training data and in situ analysis therefore go through identical code.
 */
#ifndef STEINHARDT_H
#define STEINHARDT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Default neighbor cutoff: first minimum of the LJ liquid g(r). */
#define STEINHARDT_RQ_DEFAULT 1.5

/* Number of per-atom features fed to the classifier, in this order:
 * [qbar4, qbar6, q4, q6, coord]. See steinhardt_features(). */
#define STEINHARDT_NFEAT 5

/*
 * Compute features for all resident atoms.
 *
 *   n_total     number of atoms in pos (residents + halo)
 *   pos         positions, x0 y0 z0 x1 y1 z1 ..., length 3*n_total
 *   is_resident nonzero for residents, length n_total
 *   r_q         neighbor cutoff (> 0), e.g. STEINHARDT_RQ_DEFAULT
 *   q4, q6, qbar4, qbar6   outputs, length n_total each
 *   coord       output coordination number, length n_total
 *
 * Outputs are indexed like pos. Entries for residents hold the results;
 * entries for halo atoms are set to 0. Any output pointer may be NULL if
 * that quantity is not needed.
 *
 * Cost is O(n_total) via an internal cell list (cell edge >= r_q).
 * The function is reentrant (all workspace is allocated per call).
 *
 * Returns 0 on success, -1 on bad arguments, -2 on allocation failure.
 */
int steinhardt_compute(int n_total, const double *pos, const int *is_resident,
                       double r_q,
                       double *q4, double *q6, double *qbar4, double *qbar6,
                       int *coord);

/*
 * Build a periodic image halo for a rectangular box [0,Lx)x[0,Ly)x[0,Lz).
 *
 *   n, pos   the n real atoms (wrapped into the box internally)
 *   box      box lengths (Lx, Ly, Lz)
 *   halo     halo thickness; use 2*r_q for steinhardt_compute(). Must be
 *            <= each box length (only single images are generated).
 *
 * On success returns n_ext >= n and allocates (caller frees with free()):
 *   *pos_ext     3*n_ext positions; the first n are the wrapped real atoms
 *   *is_res_ext  n_ext flags; 1 for the first n, 0 for images
 *   *orig_ext    n_ext indices of the real atom each entry is a copy of
 *                (may pass NULL if not needed)
 * Returns -1 on bad arguments, -2 on allocation failure.
 */
int steinhardt_periodic_images(int n, const double *pos, const double box[3],
                               double halo, double **pos_ext, int **is_res_ext,
                               int **orig_ext);

/*
 * Convenience wrapper: periodic images with halo 2*r_q, then
 * steinhardt_compute(). Outputs have length n (all atoms are residents).
 * Return codes as steinhardt_compute().
 */
int steinhardt_compute_periodic(int n, const double *pos, const double box[3],
                                double r_q,
                                double *q4, double *q6, double *qbar4,
                                double *qbar6, int *coord);

/*
 * Pack the classifier input vector for one atom in the canonical order
 * [qbar4, qbar6, q4, q6, coord].
 */
void steinhardt_features(double q4, double q6, double qbar4, double qbar6,
                         int coord, double feat[STEINHARDT_NFEAT]);

#ifdef __cplusplus
}
#endif

#endif /* STEINHARDT_H */
