/*
 * cluster.c: periodic cell list + union-find cluster search.
 * See cluster.h.
 */
#include "cluster.h"

#include <math.h>
#include <stdlib.h>

static int uf_find(int *par, int i)
{
    while (par[i] != i) {
        par[i] = par[par[i]]; /* path halving */
        i = par[i];
    }
    return i;
}

static void uf_union(int *par, int *sz, int i, int j)
{
    int a = uf_find(par, i), b = uf_find(par, j);
    if (a == b) return;
    if (sz[a] < sz[b] || (sz[a] == sz[b] && a > b)) {
        int t = a;
        a = b;
        b = t;
    }
    par[b] = a;
    sz[a] += sz[b];
}

int cluster_find(int n, const double *pos, const double box[3], double rc,
                 int nmin, unsigned char *in_max, cluster_info *info)
{
    if (n < 0 || (n > 0 && !pos) || !box || !info || !(rc > 0.0)) return -1;
    for (int d = 0; d < 3; d++)
        if (!(box[d] > 2.0 * rc)) return -1;
    info->nmax = info->n2 = info->nbig = info->nclus = 0;
    if (n == 0) return 0;

    int nc[3];
    for (int d = 0; d < 3; d++) {
        nc[d] = (int)floor(box[d] / rc);
        if (nc[d] < 1) nc[d] = 1;
    }
    long ncell = (long)nc[0] * nc[1] * nc[2];
    double *w = (double *)malloc((size_t)n * 3 * sizeof(double));
    int *cell = (int *)malloc((size_t)n * sizeof(int));
    int *next = (int *)malloc((size_t)n * sizeof(int));
    int *head = (int *)malloc((size_t)ncell * sizeof(int));
    int *par = (int *)malloc((size_t)n * sizeof(int));
    int *sz = (int *)malloc((size_t)n * sizeof(int));
    if (!w || !cell || !next || !head || !par || !sz) {
        free(w); free(cell); free(next); free(head); free(par); free(sz);
        return -2;
    }

    /* Wrap into [0, L) and bin */
    for (long c = 0; c < ncell; c++) head[c] = -1;
    for (int i = 0; i < n; i++) {
        int ci[3];
        for (int d = 0; d < 3; d++) {
            double L = box[d], v = pos[3 * i + d];
            v -= L * floor(v / L);
            if (v >= L) v -= L;
            w[3 * i + d] = v;
            ci[d] = (int)(v / L * nc[d]);
            if (ci[d] >= nc[d]) ci[d] = nc[d] - 1;
        }
        cell[i] = (ci[0] * nc[1] + ci[1]) * nc[2] + ci[2];
        next[i] = head[cell[i]];
        head[cell[i]] = i;
        par[i] = i;
        sz[i] = 1;
    }

    /* Bond pairs within rc (minimum image). With fewer than 3 cells in a
       direction a neighbor cell may be visited twice; union is idempotent. */
    const double rc2 = rc * rc;
    for (int i = 0; i < n; i++) {
        int c = cell[i];
        int ci[3] = {c / (nc[1] * nc[2]), (c / nc[2]) % nc[1], c % nc[2]};
        for (int a = -1; a <= 1; a++)
            for (int b = -1; b <= 1; b++)
                for (int e = -1; e <= 1; e++) {
                    int k0 = (ci[0] + a + nc[0]) % nc[0];
                    int k1 = (ci[1] + b + nc[1]) % nc[1];
                    int k2 = (ci[2] + e + nc[2]) % nc[2];
                    for (int j = head[(k0 * nc[1] + k1) * nc[2] + k2]; j >= 0; j = next[j]) {
                        if (j <= i) continue;
                        double r2 = 0.0;
                        for (int d = 0; d < 3; d++) {
                            double dx = w[3 * j + d] - w[3 * i + d];
                            dx -= box[d] * floor(dx / box[d] + 0.5);
                            r2 += dx * dx;
                        }
                        if (r2 < rc2) uf_union(par, sz, i, j);
                    }
                }
    }

    /* Cluster sizes */
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (par[i] != i) continue;
        int s = sz[i];
        info->nclus++;
        if (s >= nmin) info->nbig++;
        if (s > info->nmax) {
            info->n2 = info->nmax;
            info->nmax = s;
            best = i;
        } else if (s > info->n2) {
            info->n2 = s;
        }
    }
    if (in_max) {
        /* lowest-index root among the largest clusters */
        for (int i = 0; i < n; i++)
            if (par[i] == i && sz[i] == info->nmax) { best = i; break; }
        for (int i = 0; i < n; i++) in_max[i] = (uf_find(par, i) == best);
    }

    free(w); free(cell); free(next); free(head); free(par); free(sz);
    return 0;
}
