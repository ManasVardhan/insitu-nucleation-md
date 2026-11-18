/*
 * Unit tests for src/steinhardt.c.
 *
 *  1. Perfect FCC (open cluster, interior residents): q4, q6, qbar4, qbar6
 *     match the known FCC values, coord = 12.
 *  2. Periodic FCC box through the image-halo path: same values for every atom.
 *  3. Rotation + translation invariance on a perturbed FCC block.
 *  4. Cell-list result equals an independent O(N^2) brute-force reference
 *     (minimum image, spherical harmonics via Rodrigues + acos/atan2).
 *  5. Domain-decomposition check: residents = one subdomain of a periodic
 *     box, halo = atoms within 2*r_q; results equal the full periodic run.
 *  6. Halo-only entries are zero.
 */
#include "steinhardt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Reference FCC values (Steinhardt et al. 1983; Mickel et al. 2013). */
#define FCC_Q4 0.190940653
#define FCC_Q6 0.574524259

static int n_fail = 0, n_pass = 0;

static void check(int ok, const char *what, double val)
{
    if (ok) {
        n_pass++;
        printf("  PASS  %-58s %.3e\n", what, val);
    } else {
        n_fail++;
        printf("  FAIL  %-58s %.3e\n", what, val);
    }
}

/* Deterministic RNG (xorshift64*), so tests are reproducible everywhere. */
static unsigned long long rng_s = 88172645463325252ULL;
static double urand(void)
{
    rng_s ^= rng_s >> 12;
    rng_s ^= rng_s << 25;
    rng_s ^= rng_s >> 27;
    return (double)((rng_s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}

/* FCC lattice nx*ny*nz cells, lattice constant a. Returns atom count. */
static int make_fcc(int nx, int ny, int nz, double a, double *pos)
{
    static const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    int n = 0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                for (int b = 0; b < 4; b++) {
                    pos[3 * n] = (i + basis[b][0]) * a;
                    pos[3 * n + 1] = (j + basis[b][1]) * a;
                    pos[3 * n + 2] = (k + basis[b][2]) * a;
                    n++;
                }
    return n;
}

/* ---------- independent brute-force reference ---------- */

static double fact(int n)
{
    double f = 1;
    for (int i = 2; i <= n; i++) f *= i;
    return f;
}
static double binom(int n, int k) { return fact(n) / (fact(k) * fact(n - k)); }

/* P_l^m(x) = (-1)^m (1-x^2)^{m/2} d^m/dx^m P_l(x), P_l from its explicit sum. */
static double assoc_legendre_ref(int l, int m, double x)
{
    double d = 0.0;
    for (int k = 0; k <= l / 2; k++) {
        int p = l - 2 * k;
        if (p < m) continue;
        double c = pow(-1.0, k) * binom(l, k) * binom(2 * l - 2 * k, l) / pow(2.0, l);
        d += c * fact(p) / fact(p - m) * pow(x, p - m);
    }
    return pow(-1.0, m) * pow(1.0 - x * x, 0.5 * m) * d;
}

/* q_l of a set of bond vectors, summing over all m = -l..l explicitly. */
static void qlm_ref(int l, int nb, const double *bonds, double *re, double *im)
{
    for (int m = -l; m <= l; m++) {
        int am = m < 0 ? -m : m;
        double K = sqrt((2 * l + 1) / (4 * M_PI) * fact(l - am) / fact(l + am));
        double sr = 0, si = 0;
        for (int b = 0; b < nb; b++) {
            double x = bonds[3 * b], y = bonds[3 * b + 1], z = bonds[3 * b + 2];
            double r = sqrt(x * x + y * y + z * z);
            double th = acos(z / r), ph = atan2(y, x);
            double P = assoc_legendre_ref(l, am, cos(th));
            double yr = K * P * cos(am * ph), yi = K * P * sin(am * ph);
            if (m < 0) { /* Y_{l,-m} = (-1)^m conj(Y_lm) */
                double s = (am % 2) ? -1.0 : 1.0;
                yr *= s;
                yi *= -s;
            }
            sr += yr;
            si += yi;
        }
        re[m + l] = nb ? sr / nb : 0;
        im[m + l] = nb ? si / nb : 0;
    }
}

static double ql_ref(int l, const double *re, const double *im)
{
    double s = 0;
    for (int k = 0; k < 2 * l + 1; k++) s += re[k] * re[k] + im[k] * im[k];
    return sqrt(4 * M_PI / (2 * l + 1) * s);
}

/* O(N^2) periodic reference with minimum image. */
static void brute_periodic(int n, const double *pos, const double L[3], double rq,
                           double *q4, double *q6, double *qb4, double *qb6, int *cn)
{
    double *r4 = malloc(sizeof(double) * n * 9), *i4 = malloc(sizeof(double) * n * 9);
    double *r6 = malloc(sizeof(double) * n * 13), *i6 = malloc(sizeof(double) * n * 13);
    int *nbl = malloc(sizeof(int) * n * 64), *nnb = malloc(sizeof(int) * n);
    double *bonds = malloc(sizeof(double) * 3 * 64);
    for (int i = 0; i < n; i++) {
        int c = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double d[3];
            for (int k = 0; k < 3; k++) {
                d[k] = pos[3 * j + k] - pos[3 * i + k];
                d[k] -= L[k] * round(d[k] / L[k]);
            }
            if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < rq * rq) {
                memcpy(&bonds[3 * c], d, sizeof d);
                nbl[64 * i + c] = j;
                c++;
            }
        }
        nnb[i] = c;
        qlm_ref(4, c, bonds, &r4[9 * i], &i4[9 * i]);
        qlm_ref(6, c, bonds, &r6[13 * i], &i6[13 * i]);
    }
    for (int i = 0; i < n; i++) {
        double a4r[9], a4i[9], a6r[13], a6i[13];
        for (int k = 0; k < 9; k++) { a4r[k] = r4[9 * i + k]; a4i[k] = i4[9 * i + k]; }
        for (int k = 0; k < 13; k++) { a6r[k] = r6[13 * i + k]; a6i[k] = i6[13 * i + k]; }
        for (int c = 0; c < nnb[i]; c++) {
            int j = nbl[64 * i + c];
            for (int k = 0; k < 9; k++) { a4r[k] += r4[9 * j + k]; a4i[k] += i4[9 * j + k]; }
            for (int k = 0; k < 13; k++) { a6r[k] += r6[13 * j + k]; a6i[k] += i6[13 * j + k]; }
        }
        for (int k = 0; k < 9; k++) { a4r[k] /= nnb[i] + 1; a4i[k] /= nnb[i] + 1; }
        for (int k = 0; k < 13; k++) { a6r[k] /= nnb[i] + 1; a6i[k] /= nnb[i] + 1; }
        q4[i] = ql_ref(4, &r4[9 * i], &i4[9 * i]);
        q6[i] = ql_ref(6, &r6[13 * i], &i6[13 * i]);
        qb4[i] = ql_ref(4, a4r, a4i);
        qb6[i] = ql_ref(6, a6r, a6i);
        cn[i] = nnb[i];
    }
    free(r4); free(i4); free(r6); free(i6); free(nbl); free(nnb); free(bonds);
}

/* ---------- helpers ---------- */

typedef struct {
    double *q4, *q6, *qb4, *qb6;
    int *cn;
} feats_t;

static feats_t feats_alloc(int n)
{
    feats_t f;
    f.q4 = calloc(n, sizeof(double));
    f.q6 = calloc(n, sizeof(double));
    f.qb4 = calloc(n, sizeof(double));
    f.qb6 = calloc(n, sizeof(double));
    f.cn = calloc(n, sizeof(int));
    return f;
}
static void feats_free(feats_t *f)
{
    free(f->q4); free(f->q6); free(f->qb4); free(f->qb6); free(f->cn);
}

/* Max abs difference between two feature sets over the mask (NULL = all). */
static double feats_maxdiff(int n, const feats_t *a, const feats_t *b, const int *mask,
                            int *cn_mismatch)
{
    double m = 0;
    *cn_mismatch = 0;
    for (int i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        double d[4] = {fabs(a->q4[i] - b->q4[i]), fabs(a->q6[i] - b->q6[i]),
                       fabs(a->qb4[i] - b->qb4[i]), fabs(a->qb6[i] - b->qb6[i])};
        for (int k = 0; k < 4; k++) if (d[k] > m) m = d[k];
        if (a->cn[i] != b->cn[i]) (*cn_mismatch)++;
    }
    return m;
}

static const double RQ = STEINHARDT_RQ_DEFAULT;

/* ---------- tests ---------- */

static void test_fcc_cluster(void)
{
    printf("[1] perfect FCC, open cluster, interior residents\n");
    double a = cbrt(4.0 / 0.95);
    int nc = 8;
    double *pos = malloc(sizeof(double) * 3 * 4 * nc * nc * nc);
    int n = make_fcc(nc, nc, nc, a, pos);
    int *res = calloc(n, sizeof(int));
    int nres = 0;
    double lo = 2 * RQ + 0.01, hi = (nc - 1) * a + a / 2 - (2 * RQ + 0.01);
    for (int i = 0; i < n; i++) {
        int ok = 1;
        for (int d = 0; d < 3; d++)
            if (pos[3 * i + d] < lo || pos[3 * i + d] > hi) ok = 0;
        res[i] = ok;
        nres += ok;
    }
    feats_t f = feats_alloc(n);
    int rc = steinhardt_compute(n, pos, res, RQ, f.q4, f.q6, f.qb4, f.qb6, f.cn);
    double e4 = 0, e6 = 0, eb4 = 0, eb6 = 0;
    int bad_cn = 0;
    for (int i = 0; i < n; i++) {
        if (!res[i]) continue;
        e4 = fmax(e4, fabs(f.q4[i] - FCC_Q4));
        e6 = fmax(e6, fabs(f.q6[i] - FCC_Q6));
        eb4 = fmax(eb4, fabs(f.qb4[i] - FCC_Q4));
        eb6 = fmax(eb6, fabs(f.qb6[i] - FCC_Q6));
        if (f.cn[i] != 12) bad_cn++;
    }
    printf("  residents = %d of %d\n", nres, n);
    for (int i = 0; i < n; i++)
        if (res[i]) {
            printf("  resident %d: q4 = %.8f q6 = %.8f qbar4 = %.8f qbar6 = %.8f coord = %d\n",
                   i, f.q4[i], f.q6[i], f.qb4[i], f.qb6[i], f.cn[i]);
            break;
        }
    check(rc == 0 && nres > 0, "return code 0, residents present", (double)rc);
    check(e4 < 1e-8, "|q4 - 0.190941| max over residents", e4);
    check(e6 < 1e-8, "|q6 - 0.574524| max over residents", e6);
    check(eb4 < 1e-8, "|qbar4 - q4_fcc| max over residents", eb4);
    check(eb6 < 1e-8, "|qbar6 - q6_fcc| max over residents", eb6);
    check(bad_cn == 0, "coordination = 12 for all residents", (double)bad_cn);

    /* 6. halo-only entries zero */
    int nz = 0;
    for (int i = 0; i < n; i++)
        if (!res[i] && (f.q4[i] != 0 || f.q6[i] != 0 || f.qb4[i] != 0 || f.qb6[i] != 0 || f.cn[i] != 0))
            nz++;
    check(nz == 0, "halo-only entries are zero", (double)nz);
    feats_free(&f);
    free(pos);
    free(res);
}

static void test_fcc_periodic(void)
{
    printf("[2] perfect FCC, periodic box via image halo\n");
    const double rhos[3] = {0.90, 0.95, 1.00};
    for (int r = 0; r < 3; r++) {
        double a = cbrt(4.0 / rhos[r]);
        int nc = 5;
        double *pos = malloc(sizeof(double) * 3 * 4 * nc * nc * nc);
        int n = make_fcc(nc, nc, nc, a, pos);
        /* shift off lattice-aligned box faces to exercise wrapping */
        for (int i = 0; i < 3 * n; i++) pos[i] += 0.1 * a;
        double L[3] = {nc * a, nc * a, nc * a};
        feats_t f = feats_alloc(n);
        int rc = steinhardt_compute_periodic(n, pos, L, RQ, f.q4, f.q6, f.qb4, f.qb6, f.cn);
        double e = 0;
        int bad = 0;
        for (int i = 0; i < n; i++) {
            e = fmax(e, fabs(f.q4[i] - FCC_Q4));
            e = fmax(e, fabs(f.q6[i] - FCC_Q6));
            e = fmax(e, fabs(f.qb4[i] - FCC_Q4));
            e = fmax(e, fabs(f.qb6[i] - FCC_Q6));
            if (f.cn[i] != 12) bad++;
        }
        char msg[128];
        snprintf(msg, sizeof msg, "rho=%.2f N=%d: all atoms q/qbar = FCC values", rhos[r], n);
        check(rc == 0 && e < 1e-8, msg, e);
        snprintf(msg, sizeof msg, "rho=%.2f: coordination 12 for all atoms", rhos[r]);
        check(bad == 0, msg, (double)bad);
        feats_free(&f);
        free(pos);
    }
}

static void rand_rotation(double R[3][3])
{
    /* random unit quaternion */
    double u1 = urand(), u2 = urand(), u3 = urand();
    double q0 = sqrt(1 - u1) * sin(2 * M_PI * u2), q1 = sqrt(1 - u1) * cos(2 * M_PI * u2);
    double q2 = sqrt(u1) * sin(2 * M_PI * u3), q3 = sqrt(u1) * cos(2 * M_PI * u3);
    R[0][0] = 1 - 2 * (q2 * q2 + q3 * q3); R[0][1] = 2 * (q1 * q2 - q0 * q3); R[0][2] = 2 * (q1 * q3 + q0 * q2);
    R[1][0] = 2 * (q1 * q2 + q0 * q3); R[1][1] = 1 - 2 * (q1 * q1 + q3 * q3); R[1][2] = 2 * (q2 * q3 - q0 * q1);
    R[2][0] = 2 * (q1 * q3 - q0 * q2); R[2][1] = 2 * (q2 * q3 + q0 * q1); R[2][2] = 1 - 2 * (q1 * q1 + q2 * q2);
}

static void test_invariance(void)
{
    printf("[3] rotation and translation invariance (perturbed FCC + liquid-like)\n");
    double a = cbrt(4.0 / 0.95);
    int nc = 7;
    double *pos = malloc(sizeof(double) * 3 * 4 * nc * nc * nc);
    int n = make_fcc(nc, nc, nc, a, pos);
    for (int i = 0; i < 3 * n; i++) pos[i] += 0.15 * (urand() - 0.5) * 2; /* +-0.15 */
    double c[3] = {0, 0, 0};
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) c[d] += pos[3 * i + d] / n;
    int *res = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        double r2 = 0;
        for (int d = 0; d < 3; d++) r2 += (pos[3 * i + d] - c[d]) * (pos[3 * i + d] - c[d]);
        res[i] = sqrt(r2) < 0.5 * nc * a - 2 * RQ - 0.3;
    }
    feats_t f0 = feats_alloc(n);
    steinhardt_compute(n, pos, res, RQ, f0.q4, f0.q6, f0.qb4, f0.qb6, f0.cn);

    double *p2 = malloc(sizeof(double) * 3 * n);
    double worst = 0;
    int worst_cn = 0;
    for (int trial = 0; trial < 5; trial++) {
        double R[3][3], t[3] = {100 * (urand() - 0.5), 100 * (urand() - 0.5), 100 * (urand() - 0.5)};
        rand_rotation(R);
        for (int i = 0; i < n; i++)
            for (int d = 0; d < 3; d++)
                p2[3 * i + d] = R[d][0] * pos[3 * i] + R[d][1] * pos[3 * i + 1] + R[d][2] * pos[3 * i + 2] + t[d];
        feats_t f1 = feats_alloc(n);
        steinhardt_compute(n, p2, res, RQ, f1.q4, f1.q6, f1.qb4, f1.qb6, f1.cn);
        int cnm;
        double d = feats_maxdiff(n, &f0, &f1, res, &cnm);
        if (d > worst) worst = d;
        worst_cn += cnm;
        feats_free(&f1);
    }
    /* pure translation */
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) p2[3 * i + d] = pos[3 * i + d] - 37.25 + d;
    feats_t f2 = feats_alloc(n);
    steinhardt_compute(n, p2, res, RQ, f2.q4, f2.q6, f2.qb4, f2.qb6, f2.cn);
    int cnt;
    double dt = feats_maxdiff(n, &f0, &f2, res, &cnt);
    check(worst < 1e-10 && worst_cn == 0, "5 random rotations+translations: max |diff|", worst);
    check(dt < 1e-10 && cnt == 0, "pure translation: max |diff|", dt);
    /* permutation invariance (reverse order) */
    int *res_r = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) p2[3 * i + d] = pos[3 * (n - 1 - i) + d];
        res_r[i] = res[n - 1 - i];
    }
    feats_t f3 = feats_alloc(n);
    steinhardt_compute(n, p2, res_r, RQ, f3.q4, f3.q6, f3.qb4, f3.qb6, f3.cn);
    double dp = 0;
    for (int i = 0; i < n; i++) {
        if (!res[i]) continue;
        int j = n - 1 - i;
        dp = fmax(dp, fabs(f0.qb6[i] - f3.qb6[j]) + fabs(f0.q4[i] - f3.q4[j]));
    }
    check(dp < 1e-12, "atom reordering: max |diff|", dp);
    feats_free(&f0); feats_free(&f2); feats_free(&f3);
    free(p2); free(pos); free(res); free(res_r);
}

static void test_bruteforce(void)
{
    printf("[4] cell list vs O(N^2) reference (independent Ylm), periodic\n");
    const double amp[2] = {0.10, 0.45}; /* thermal-crystal-like, liquid-like */
    for (int t = 0; t < 2; t++) {
        double a = cbrt(4.0 / 0.95);
        int nc = 5;
        double *pos = malloc(sizeof(double) * 3 * 4 * nc * nc * nc);
        int n = make_fcc(nc, nc, nc, a, pos);
        for (int i = 0; i < 3 * n; i++) pos[i] += amp[t] * 2 * (urand() - 0.5);
        double L[3] = {nc * a, nc * a, nc * a};
        feats_t f = feats_alloc(n), g = feats_alloc(n);
        steinhardt_compute_periodic(n, pos, L, RQ, f.q4, f.q6, f.qb4, f.qb6, f.cn);
        brute_periodic(n, pos, L, RQ, g.q4, g.q6, g.qb4, g.qb6, g.cn);
        int cnm;
        double d = feats_maxdiff(n, &f, &g, NULL, &cnm);
        char msg[128];
        snprintf(msg, sizeof msg, "perturbation +-%.2f, N=%d: max |diff|", amp[t], n);
        check(d < 1e-10 && cnm == 0, msg, d);
        feats_free(&f); feats_free(&g);
        free(pos);
    }
}

static void test_subdomain(void)
{
    printf("[5] subdomain residents + 2*r_q halo == full periodic result\n");
    double a = cbrt(4.0 / 0.95);
    int nc = 7;
    double *pos = malloc(sizeof(double) * 3 * 4 * nc * nc * nc);
    int n = make_fcc(nc, nc, nc, a, pos);
    for (int i = 0; i < 3 * n; i++) pos[i] += 0.4 * 2 * (urand() - 0.5);
    double L[3] = {nc * a, nc * a, nc * a};
    feats_t full = feats_alloc(n);
    steinhardt_compute_periodic(n, pos, L, RQ, full.q4, full.q6, full.qb4, full.qb6, full.cn);

    /* Image-extended set, then emulate one rank owning octant [0,L/2)^3. */
    double *pe;
    int *re, *oe;
    int ne = steinhardt_periodic_images(n, pos, L, 2 * RQ, &pe, &re, &oe);
    double halo = 2 * RQ;
    double *ps = malloc(sizeof(double) * 3 * ne);
    int *rs = malloc(sizeof(int) * ne), *os = malloc(sizeof(int) * ne);
    int ns = 0, nres = 0;
    for (int k = 0; k < ne; k++) {
        int inside = 1, near = 1;
        for (int d = 0; d < 3; d++) {
            double v = pe[3 * k + d];
            if (!(v >= 0 && v < L[d] / 2)) inside = 0;
            if (!(v >= -halo && v < L[d] / 2 + halo)) near = 0;
        }
        if (!near) continue;
        memcpy(&ps[3 * ns], &pe[3 * k], 3 * sizeof(double));
        rs[ns] = inside && re[k]; /* resident only if a real atom in the octant */
        os[ns] = oe[k];
        nres += rs[ns];
        ns++;
    }
    feats_t sub = feats_alloc(ns);
    steinhardt_compute(ns, ps, rs, RQ, sub.q4, sub.q6, sub.qb4, sub.qb6, sub.cn);
    double m = 0;
    int cnm = 0;
    for (int k = 0; k < ns; k++) {
        if (!rs[k]) continue;
        int i = os[k];
        m = fmax(m, fabs(sub.q4[k] - full.q4[i]));
        m = fmax(m, fabs(sub.q6[k] - full.q6[i]));
        m = fmax(m, fabs(sub.qb4[k] - full.qb4[i]));
        m = fmax(m, fabs(sub.qb6[k] - full.qb6[i]));
        if (sub.cn[k] != full.cn[i]) cnm++;
    }
    printf("  residents = %d, halo atoms = %d\n", nres, ns - nres);
    check(nres > 0 && m < 1e-12 && cnm == 0, "octant residents vs full box: max |diff|", m);

    /* Show the requirement is real: a 1*r_q halo gives wrong qbar at edges. */
    int ns2 = 0;
    for (int k = 0; k < ns; k++) {
        int near = 1;
        for (int d = 0; d < 3; d++) {
            double v = ps[3 * k + d];
            if (!(v >= -RQ && v < L[d] / 2 + RQ)) near = 0;
        }
        if (!near) continue;
        memcpy(&pe[3 * ns2], &ps[3 * k], 3 * sizeof(double));
        re[ns2] = rs[k];
        oe[ns2] = os[k];
        ns2++;
    }
    feats_t thin = feats_alloc(ns2);
    steinhardt_compute(ns2, pe, re, RQ, thin.q4, thin.q6, thin.qb4, thin.qb6, thin.cn);
    double mq = 0, mqb = 0;
    for (int k = 0; k < ns2; k++) {
        if (!re[k]) continue;
        int i = oe[k];
        mq = fmax(mq, fabs(thin.q6[k] - full.q6[i]));
        mqb = fmax(mqb, fabs(thin.qb6[k] - full.qb6[i]));
    }
    printf("  (info) r_q-thick halo: max |dq6| = %.2e (exact), max |dqbar6| = %.2e (wrong)\n", mq, mqb);
    check(mq < 1e-12 && mqb > 1e-6, "thin halo: q exact but qbar wrong (requirement holds)", mqb);
    feats_free(&full); feats_free(&sub); feats_free(&thin);
    free(pos); free(pe); free(re); free(oe); free(ps); free(rs); free(os);
}

int main(void)
{
    test_fcc_cluster();
    test_fcc_periodic();
    test_invariance();
    test_bruteforce();
    test_subdomain();
    printf("steinhardt tests: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
