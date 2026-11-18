/*
 * steinhardt.c: Steinhardt q4/q6 and Lechner-Dellago qbar4/qbar6.
 * See steinhardt.h for definitions, API, and the halo requirement.
 *
 * Spherical harmonics: for m >= 0,
 *   Y_lm(r) = K_lm * Pt_l^m(z/r) * ((x + i y)/r)^m,
 *   K_lm    = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! ),
 * where Pt_l^m is the polynomial part of the associated Legendre function,
 * P_l^m(c) = (1-c^2)^{m/2} Pt_l^m(c) (Condon-Shortley phase included).
 * This form has no singularity at the poles. Negative m are not stored:
 * Y_{l,-m} = (-1)^m conj(Y_lm), hence sum_{m=-l..l} |q_lm|^2 =
 * |q_l0|^2 + 2 sum_{m>0} |q_lm|^2, which also holds for the averages.
 */
#include "steinhardt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Per-atom storage: complex q_lm for l=4 (m=0..4) and l=6 (m=0..6). */
#define NQ4 5
#define NQ6 7
#define NQLM (2 * (NQ4 + NQ6)) /* 24 doubles: re,im pairs */

typedef struct {
    double k4[NQ4];
    double k6[NQ6];
} ylm_norm_t;

static double factorial(int n)
{
    double f = 1.0;
    for (int i = 2; i <= n; i++) f *= (double)i;
    return f;
}

static void ylm_norm_init(ylm_norm_t *kn)
{
    for (int m = 0; m <= 4; m++)
        kn->k4[m] = sqrt((2 * 4 + 1) / (4.0 * M_PI) * factorial(4 - m) / factorial(4 + m));
    for (int m = 0; m <= 6; m++)
        kn->k6[m] = sqrt((2 * 6 + 1) / (4.0 * M_PI) * factorial(6 - m) / factorial(6 + m));
}

/* Polynomial part Pt_l^m(c) for given l, m via upward recurrence in l. */
static double legendre_poly_part(int l, int m, double c)
{
    double pmm = 1.0; /* Pt_m^m = (-1)^m (2m-1)!! */
    for (int k = 1; k <= m; k++) pmm *= -(2.0 * k - 1.0);
    if (l == m) return pmm;
    double pm1 = c * (2.0 * m + 1.0) * pmm; /* Pt_{m+1}^m */
    if (l == m + 1) return pm1;
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ll++) {
        pll = (c * (2.0 * ll - 1.0) * pm1 - (ll + m - 1.0) * pmm) / (double)(ll - m);
        pmm = pm1;
        pm1 = pll;
    }
    return pll;
}

/* Accumulate Y_4m and Y_6m (m >= 0) of bond vector (dx,dy,dz), |d| = r > 0. */
static void ylm_accumulate(const ylm_norm_t *kn, double dx, double dy, double dz,
                           double r, double *acc)
{
    double inv = 1.0 / r;
    double c = dz * inv;
    double ex = dx * inv, ey = dy * inv; /* (x+iy)/r */
    double pre = 1.0, pim = 0.0;         /* ((x+iy)/r)^m */
    for (int m = 0; m <= 6; m++) {
        if (m <= 4) {
            double a = kn->k4[m] * legendre_poly_part(4, m, c);
            acc[2 * m] += a * pre;
            acc[2 * m + 1] += a * pim;
        }
        double b = kn->k6[m] * legendre_poly_part(6, m, c);
        acc[2 * NQ4 + 2 * m] += b * pre;
        acc[2 * NQ4 + 2 * m + 1] += b * pim;
        double nre = pre * ex - pim * ey;
        double nim = pre * ey + pim * ex;
        pre = nre;
        pim = nim;
    }
}

/* q_l from stored m >= 0 components. */
static double ql_from_qlm(const double *qlm, int l)
{
    int nm = l + 1;
    double s = qlm[0] * qlm[0] + qlm[1] * qlm[1];
    for (int m = 1; m < nm; m++)
        s += 2.0 * (qlm[2 * m] * qlm[2 * m] + qlm[2 * m + 1] * qlm[2 * m + 1]);
    return sqrt(4.0 * M_PI / (2.0 * l + 1.0) * s);
}

/* ---------------- cell list (non-periodic) ---------------- */

typedef struct {
    int nc[3];
    double lo[3];
    double inv_w[3];
    int *head; /* nc0*nc1*nc2 */
    int *next; /* n */
} cells_t;

static int cells_build(cells_t *cl, int n, const double *pos, double r_q)
{
    double lo[3] = {pos[0], pos[1], pos[2]}, hi[3] = {pos[0], pos[1], pos[2]};
    for (int i = 1; i < n; i++)
        for (int d = 0; d < 3; d++) {
            double v = pos[3 * i + d];
            if (v < lo[d]) lo[d] = v;
            if (v > hi[d]) hi[d] = v;
        }
    /* Cell edge >= w; grow w if the grid would be absurdly large relative to n
     * (e.g. very sparse or widely scattered input). */
    double w = r_q;
    long max_cells = 8L * n + 64;
    for (;;) {
        long tot = 1;
        for (int d = 0; d < 3; d++) {
            long k = (long)floor((hi[d] - lo[d]) / w);
            if (k < 1) k = 1;
            cl->nc[d] = (int)k;
            tot *= k;
        }
        if (tot <= max_cells) break;
        w *= 1.26;
    }
    for (int d = 0; d < 3; d++) {
        cl->lo[d] = lo[d];
        double ext = hi[d] - lo[d];
        cl->inv_w[d] = (ext > 0.0) ? cl->nc[d] / ext : 0.0;
    }
    long ncell = (long)cl->nc[0] * cl->nc[1] * cl->nc[2];
    cl->head = (int *)malloc((size_t)ncell * sizeof(int));
    cl->next = (int *)malloc((size_t)n * sizeof(int));
    if (!cl->head || !cl->next) {
        free(cl->head);
        free(cl->next);
        return -2;
    }
    for (long c = 0; c < ncell; c++) cl->head[c] = -1;
    for (int i = 0; i < n; i++) {
        int ci[3];
        for (int d = 0; d < 3; d++) {
            int k = (int)((pos[3 * i + d] - cl->lo[d]) * cl->inv_w[d]);
            if (k < 0) k = 0;
            if (k >= cl->nc[d]) k = cl->nc[d] - 1;
            ci[d] = k;
        }
        long c = ((long)ci[0] * cl->nc[1] + ci[1]) * cl->nc[2] + ci[2];
        cl->next[i] = cl->head[c];
        cl->head[c] = i;
    }
    return 0;
}

static void cells_free(cells_t *cl)
{
    free(cl->head);
    free(cl->next);
}

static void cell_of(const cells_t *cl, const double *p, int ci[3])
{
    for (int d = 0; d < 3; d++) {
        int k = (int)((p[d] - cl->lo[d]) * cl->inv_w[d]);
        if (k < 0) k = 0;
        if (k >= cl->nc[d]) k = cl->nc[d] - 1;
        ci[d] = k;
    }
}

/* Iterate over neighbors j != i of atom i within r_q. The body sees j, dx,
 * dy, dz (r_j - r_i) and r2. The body is variadic so it may contain commas. */
#define FOR_NEIGHBORS(cl, pos, i, rq2, ...)                                    \
    do {                                                                       \
        int ci_[3];                                                            \
        const double *pi_ = &(pos)[3 * (i)];                                   \
        cell_of((cl), pi_, ci_);                                               \
        for (int a_ = ci_[0] - 1; a_ <= ci_[0] + 1; a_++) {                    \
            if (a_ < 0 || a_ >= (cl)->nc[0]) continue;                         \
            for (int b_ = ci_[1] - 1; b_ <= ci_[1] + 1; b_++) {                \
                if (b_ < 0 || b_ >= (cl)->nc[1]) continue;                     \
                for (int c_ = ci_[2] - 1; c_ <= ci_[2] + 1; c_++) {            \
                    if (c_ < 0 || c_ >= (cl)->nc[2]) continue;                 \
                    long cc_ = ((long)a_ * (cl)->nc[1] + b_) * (cl)->nc[2] + c_; \
                    for (int j = (cl)->head[cc_]; j >= 0; j = (cl)->next[j]) { \
                        if (j == (i)) continue;                                \
                        double dx = (pos)[3 * j] - pi_[0];                     \
                        double dy = (pos)[3 * j + 1] - pi_[1];                 \
                        double dz = (pos)[3 * j + 2] - pi_[2];                 \
                        double r2 = dx * dx + dy * dy + dz * dz;               \
                        if (r2 >= (rq2)) continue;                             \
                        __VA_ARGS__                                            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

int steinhardt_compute(int n_total, const double *pos, const int *is_resident,
                       double r_q,
                       double *q4, double *q6, double *qbar4, double *qbar6,
                       int *coord)
{
    if (n_total < 0 || (n_total > 0 && (!pos || !is_resident)) || !(r_q > 0.0))
        return -1;

    /* Zero outputs (halo entries stay 0). */
    for (int i = 0; i < n_total; i++) {
        if (q4) q4[i] = 0.0;
        if (q6) q6[i] = 0.0;
        if (qbar4) qbar4[i] = 0.0;
        if (qbar6) qbar6[i] = 0.0;
        if (coord) coord[i] = 0;
    }
    if (n_total == 0) return 0;

    ylm_norm_t kn;
    ylm_norm_init(&kn);
    const double rq2 = r_q * r_q;

    cells_t cl;
    if (cells_build(&cl, n_total, pos, r_q) != 0) return -2;

    /* needed[i]: 1 if q_lm(i) is required (resident or neighbor of one). */
    unsigned char *needed = (unsigned char *)calloc((size_t)n_total, 1);
    double *qlm = (double *)calloc((size_t)n_total * NQLM, sizeof(double));
    int *nb = (int *)calloc((size_t)n_total, sizeof(int));
    if (!needed || !qlm || !nb) {
        free(needed);
        free(qlm);
        free(nb);
        cells_free(&cl);
        return -2;
    }

    /* Pass 1: mark residents and their neighbors. */
    for (int i = 0; i < n_total; i++) {
        if (!is_resident[i]) continue;
        needed[i] = 1;
        FOR_NEIGHBORS(&cl, pos, i, rq2, { needed[j] = 1; (void)dx; (void)dy; (void)dz; });
    }

    /* Pass 2: q_lm for needed atoms. Passes 2 and 3 write only entry i, so
     * they parallelize with OpenMP (if compiled with -fopenmp) without
     * changing any result bit. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 512)
#endif
    for (int i = 0; i < n_total; i++) {
        if (!needed[i]) continue;
        double *acc = &qlm[(size_t)i * NQLM];
        int cnt = 0;
        FOR_NEIGHBORS(&cl, pos, i, rq2, {
            ylm_accumulate(&kn, dx, dy, dz, sqrt(r2), acc);
            cnt++;
        });
        nb[i] = cnt;
        if (cnt > 0) {
            double inv = 1.0 / cnt;
            for (int k = 0; k < NQLM; k++) acc[k] *= inv;
        }
    }

    /* Pass 3: q_l and Lechner-Dellago averages for residents. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 512)
#endif
    for (int i = 0; i < n_total; i++) {
        if (!is_resident[i]) continue;
        const double *qi = &qlm[(size_t)i * NQLM];
        double avg[NQLM];
        memcpy(avg, qi, sizeof(avg));
        FOR_NEIGHBORS(&cl, pos, i, rq2, {
            const double *qj = &qlm[(size_t)j * NQLM];
            for (int k = 0; k < NQLM; k++) avg[k] += qj[k];
            (void)dx; (void)dy; (void)dz;
        });
        double inv = 1.0 / (nb[i] + 1);
        for (int k = 0; k < NQLM; k++) avg[k] *= inv;

        if (q4) q4[i] = ql_from_qlm(qi, 4);
        if (q6) q6[i] = ql_from_qlm(qi + 2 * NQ4, 6);
        if (qbar4) qbar4[i] = ql_from_qlm(avg, 4);
        if (qbar6) qbar6[i] = ql_from_qlm(avg + 2 * NQ4, 6);
        if (coord) coord[i] = nb[i];
    }

    free(needed);
    free(qlm);
    free(nb);
    cells_free(&cl);
    return 0;
}

int steinhardt_periodic_images(int n, const double *pos, const double box[3],
                               double halo, double **pos_ext, int **is_res_ext,
                               int **orig_ext)
{
    if (n < 0 || (n > 0 && !pos) || !box || !pos_ext || !is_res_ext || halo < 0.0)
        return -1;
    for (int d = 0; d < 3; d++)
        if (!(box[d] > 0.0) || halo > box[d]) return -1;

    /* Wrap real atoms into [0, L). */
    double *w = (double *)malloc((size_t)(n > 0 ? n : 1) * 3 * sizeof(double));
    if (!w) return -2;
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) {
            double L = box[d];
            double v = pos[3 * i + d];
            v -= L * floor(v / L);
            if (v >= L) v -= L; /* guard against rounding up to L */
            w[3 * i + d] = v;
        }

    /* Count images first. */
    long n_ext = n;
    for (int pass = 0; pass < 2; pass++) {
        double *pe = NULL;
        int *re = NULL, *oe = NULL;
        if (pass == 1) {
            pe = (double *)malloc((size_t)n_ext * 3 * sizeof(double));
            re = (int *)malloc((size_t)n_ext * sizeof(int));
            oe = (int *)malloc((size_t)n_ext * sizeof(int));
            if (!pe || !re || !oe) {
                free(pe);
                free(re);
                free(oe);
                free(w);
                return -2;
            }
            for (int i = 0; i < n; i++) {
                pe[3 * i] = w[3 * i];
                pe[3 * i + 1] = w[3 * i + 1];
                pe[3 * i + 2] = w[3 * i + 2];
                re[i] = 1;
                oe[i] = i;
            }
        }
        long k = n;
        for (int i = 0; i < n; i++) {
            /* Allowed shifts per dimension: 0 always; +1 if near low face
             * (image appears beyond high face), -1 if near high face. */
            int sh[3][3], ns[3];
            for (int d = 0; d < 3; d++) {
                ns[d] = 0;
                sh[d][ns[d]++] = 0;
                if (w[3 * i + d] < halo) sh[d][ns[d]++] = 1;
                if (w[3 * i + d] >= box[d] - halo) sh[d][ns[d]++] = -1;
            }
            for (int a = 0; a < ns[0]; a++)
                for (int b = 0; b < ns[1]; b++)
                    for (int c = 0; c < ns[2]; c++) {
                        if (sh[0][a] == 0 && sh[1][b] == 0 && sh[2][c] == 0) continue;
                        if (pass == 1) {
                            pe[3 * k] = w[3 * i] + sh[0][a] * box[0];
                            pe[3 * k + 1] = w[3 * i + 1] + sh[1][b] * box[1];
                            pe[3 * k + 2] = w[3 * i + 2] + sh[2][c] * box[2];
                            re[k] = 0;
                            oe[k] = i;
                        }
                        k++;
                    }
        }
        if (pass == 0) {
            n_ext = k;
            if (n_ext > 2147483647L) {
                free(w);
                return -2;
            }
        } else {
            *pos_ext = pe;
            *is_res_ext = re;
            if (orig_ext)
                *orig_ext = oe;
            else
                free(oe);
        }
    }
    free(w);
    return (int)n_ext;
}

int steinhardt_compute_periodic(int n, const double *pos, const double box[3],
                                double r_q,
                                double *q4, double *q6, double *qbar4,
                                double *qbar6, int *coord)
{
    if (!(r_q > 0.0)) return -1;
    double *pe = NULL;
    int *re = NULL;
    int n_ext = steinhardt_periodic_images(n, pos, box, 2.0 * r_q, &pe, &re, NULL);
    if (n_ext < 0) return n_ext;

    double *t4 = (double *)malloc((size_t)n_ext * sizeof(double));
    double *t6 = (double *)malloc((size_t)n_ext * sizeof(double));
    double *tb4 = (double *)malloc((size_t)n_ext * sizeof(double));
    double *tb6 = (double *)malloc((size_t)n_ext * sizeof(double));
    int *tc = (int *)malloc((size_t)n_ext * sizeof(int));
    int rc = -2;
    if (t4 && t6 && tb4 && tb6 && tc) {
        rc = steinhardt_compute(n_ext, pe, re, r_q, t4, t6, tb4, tb6, tc);
        if (rc == 0)
            for (int i = 0; i < n; i++) {
                if (q4) q4[i] = t4[i];
                if (q6) q6[i] = t6[i];
                if (qbar4) qbar4[i] = tb4[i];
                if (qbar6) qbar6[i] = tb6[i];
                if (coord) coord[i] = tc[i];
            }
    }
    free(t4);
    free(t6);
    free(tb4);
    free(tb6);
    free(tc);
    free(pe);
    free(re);
    return rc;
}

void steinhardt_features(double q4, double q6, double qbar4, double qbar6,
                         int coord, double feat[STEINHARDT_NFEAT])
{
    feat[0] = qbar4;
    feat[1] = qbar6;
    feat[2] = q4;
    feat[3] = q6;
    feat[4] = (double)coord;
}
