/*
 * cluster.h: connected clusters of (crystalline) atoms in a periodic
 * rectangular box. Two atoms are bonded if their minimum-image distance is
 * below rc. Cell list + union-find, O(n). Plain C99, no MPI.
 *
 * Used by the in situ ANA root (insitu.c) and by the post hoc check
 * (posthoc.c), so both find clusters with identical code.
 */
#ifndef CLUSTER_H
#define CLUSTER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nmax;     /* size of the largest cluster (0 if n = 0) */
    int n2;       /* size of the second largest cluster */
    int nbig;     /* # of clusters with size >= nmin */
    int nclus;    /* total # of clusters */
} cluster_info;

/*
 * n atoms, pos[3*n] (any image; wrapped internally), box[3] lengths,
 * rc bond cutoff (< box/2 in each direction), nmin for nbig.
 * in_max (length n, may be NULL): set to 1 for atoms of the largest
 * cluster, 0 otherwise. Ties between equal-size largest clusters go to the
 * one containing the lowest index.
 * Returns 0 on success, -1 on bad arguments, -2 on allocation failure.
 */
int cluster_find(int n, const double *pos, const double box[3], double rc,
                 int nmin, unsigned char *in_max, cluster_info *info);

#ifdef __cplusplus
}
#endif

#endif /* CLUSTER_H */
