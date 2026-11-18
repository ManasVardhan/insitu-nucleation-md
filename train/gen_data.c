/*
 * gen_data.c: serial Lennard-Jones MD that generates labeled training and
 * evaluation snapshots for the liquid/crystal classifier.
 *
 * Model: LJ, reduced units, r_c = 2.5 (truncated forces, energy shifted for
 * reporting), velocity Verlet, dt = 0.005, PBC, linked-list cells,
 * velocity-rescaling thermostat applied every step.
 *
 * Per-atom features come from src/steinhardt.c through the periodic
 * image-halo path, i.e. exactly the function that runs in situ.
 *
 * Usage:
 *   gen_data crystal T rho ncell nsnap seed   > out.csv
 *   gen_data liquid  T rho ncell nsnap seed   > out.csv
 *   gen_data slab    T rho ncell nsnap seed   > out.csv
 *     ncell: FCC unit cells per side (slab: ncell x ncell x 2*ncell)
 *
 * Protocols:
 *   crystal: FCC at rho, velocities at T, equilibrate EQ_STEPS, then nsnap
 *            snapshots SNAP_EVERY apart. Label 1.
 *   liquid:  FCC, melt at T_MELT for MELT_STEPS, instant quench to T,
 *            equilibrate EQ_STEPS, then nsnap snapshots. Label 0.
 *   slab:    thermalize FCC at T, freeze the middle half along z (positions
 *            and saved velocities), melt the rest at T_MELT, quench the
 *            liquid to T for SLAB_QUENCH_STEPS, then release everything at T
 *            and take nsnap snapshots SLAB_SNAP_EVERY apart. Label = 1 for
 *            atoms that started in the frozen half. "dist" = signed distance
 *            of the atom's current z from the nearest initial interface plane
 *            (positive on the crystal side).
 *
 * Output CSV (stdout):
 *   set,T,rho,snap,id,label,z,dist,qbar4,qbar6,q4,q6,coord
 * Diagnostics (stderr): per snapshot T, PE/atom, mean qbar6, fraction of
 * atoms with qbar6 > 0.35 (crystal-like).
 */
#include "steinhardt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RC 2.5
#define DT 0.005
#define T_MELT 2.0
#define MELT_STEPS 3000
#define EQ_STEPS 2000
#define SNAP_EVERY 500
#define SLAB_THERM_STEPS 1000
#define SLAB_QUENCH_STEPS 200
#define SLAB_SNAP_EVERY 50
#define QBAR6_XTAL 0.35

typedef struct {
    int n;
    double L[3];
    double *r, *v, *f;
    unsigned char *frozen; /* frozen atoms are not integrated */
    /* cell list */
    int nc[3];
    int *head, *next;
    double pe; /* potential energy (shifted) */
} sys_t;

/* ---------- RNG ---------- */
static unsigned long long rng_s = 1;
static double urand(void)
{
    rng_s ^= rng_s >> 12;
    rng_s ^= rng_s << 25;
    rng_s ^= rng_s >> 27;
    return ((double)((rng_s * 2685821657736338717ULL) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double gauss(void)
{
    return sqrt(-2.0 * log(urand())) * cos(2.0 * M_PI * urand());
}

/* ---------- setup ---------- */
static void fcc_init(sys_t *s, int nx, int ny, int nz, double rho)
{
    static const double basis[4][3] = {{0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                                       {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
    double a = cbrt(4.0 / rho);
    s->n = 4 * nx * ny * nz;
    s->L[0] = nx * a;
    s->L[1] = ny * a;
    s->L[2] = nz * a;
    s->r = malloc(sizeof(double) * 3 * s->n);
    s->v = calloc(3 * s->n, sizeof(double));
    s->f = calloc(3 * s->n, sizeof(double));
    s->frozen = calloc(s->n, 1);
    int k = 0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int l = 0; l < nz; l++)
                for (int b = 0; b < 4; b++) {
                    s->r[3 * k] = (i + basis[b][0]) * a;
                    s->r[3 * k + 1] = (j + basis[b][1]) * a;
                    s->r[3 * k + 2] = (l + basis[b][2]) * a;
                    k++;
                }
    for (int d = 0; d < 3; d++) {
        s->nc[d] = (int)floor(s->L[d] / RC);
        if (s->nc[d] < 3) {
            fprintf(stderr, "box too small for cell list (need >= 3 cells)\n");
            exit(1);
        }
    }
    s->head = malloc(sizeof(int) * s->nc[0] * s->nc[1] * s->nc[2]);
    s->next = malloc(sizeof(int) * s->n);
}

static double temperature(const sys_t *s)
{
    double ke = 0;
    int nm = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->frozen[i]) continue;
        nm++;
        for (int d = 0; d < 3; d++) ke += s->v[3 * i + d] * s->v[3 * i + d];
    }
    return nm ? ke / (3.0 * nm) : 0.0;
}

static void rescale(sys_t *s, double T)
{
    double Tc = temperature(s);
    if (Tc <= 0) return;
    double f = sqrt(T / Tc);
    for (int i = 0; i < s->n; i++)
        if (!s->frozen[i])
            for (int d = 0; d < 3; d++) s->v[3 * i + d] *= f;
}

static void init_velocities(sys_t *s, double T)
{
    double vc[3] = {0, 0, 0};
    int nm = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->frozen[i]) continue;
        nm++;
        for (int d = 0; d < 3; d++) {
            s->v[3 * i + d] = gauss();
            vc[d] += s->v[3 * i + d];
        }
    }
    for (int i = 0; i < s->n; i++)
        if (!s->frozen[i])
            for (int d = 0; d < 3; d++) s->v[3 * i + d] -= vc[d] / nm;
    rescale(s, T);
}

/* ---------- forces ---------- */
static void forces(sys_t *s)
{
    const int n = s->n;
    const double rc2 = RC * RC;
    const double ecut = 4.0 * (pow(RC, -12) - pow(RC, -6));
    int ncell = s->nc[0] * s->nc[1] * s->nc[2];
    for (int c = 0; c < ncell; c++) s->head[c] = -1;
    for (int i = 0; i < n; i++) {
        int ci[3];
        for (int d = 0; d < 3; d++) {
            double x = s->r[3 * i + d];
            x -= s->L[d] * floor(x / s->L[d]);
            s->r[3 * i + d] = x;
            int k = (int)(x / s->L[d] * s->nc[d]);
            if (k >= s->nc[d]) k = s->nc[d] - 1;
            ci[d] = k;
        }
        int c = (ci[0] * s->nc[1] + ci[1]) * s->nc[2] + ci[2];
        s->next[i] = s->head[c];
        s->head[c] = i;
    }
    memset(s->f, 0, sizeof(double) * 3 * n);
    double pe = 0;
    for (int cx = 0; cx < s->nc[0]; cx++)
        for (int cy = 0; cy < s->nc[1]; cy++)
            for (int cz = 0; cz < s->nc[2]; cz++) {
                int c = (cx * s->nc[1] + cy) * s->nc[2] + cz;
                for (int ax = -1; ax <= 1; ax++)
                    for (int ay = -1; ay <= 1; ay++)
                        for (int az = -1; az <= 1; az++) {
                            int nx = (cx + ax + s->nc[0]) % s->nc[0];
                            int ny = (cy + ay + s->nc[1]) % s->nc[1];
                            int nz = (cz + az + s->nc[2]) % s->nc[2];
                            int c2 = (nx * s->nc[1] + ny) * s->nc[2] + nz;
                            for (int i = s->head[c]; i >= 0; i = s->next[i])
                                for (int j = s->head[c2]; j >= 0; j = s->next[j]) {
                                    if (j <= i) continue;
                                    double d[3], r2 = 0;
                                    for (int k = 0; k < 3; k++) {
                                        d[k] = s->r[3 * i + k] - s->r[3 * j + k];
                                        if (d[k] > 0.5 * s->L[k]) d[k] -= s->L[k];
                                        else if (d[k] < -0.5 * s->L[k]) d[k] += s->L[k];
                                        r2 += d[k] * d[k];
                                    }
                                    if (r2 >= rc2) continue;
                                    double ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
                                    double fr = 24.0 * ir6 * (2.0 * ir6 - 1.0) * ir2;
                                    pe += 4.0 * ir6 * (ir6 - 1.0) - ecut;
                                    for (int k = 0; k < 3; k++) {
                                        s->f[3 * i + k] += fr * d[k];
                                        s->f[3 * j + k] -= fr * d[k];
                                    }
                                }
                        }
            }
    s->pe = pe;
}

static void step(sys_t *s, double T)
{
    for (int i = 0; i < s->n; i++) {
        if (s->frozen[i]) continue;
        for (int d = 0; d < 3; d++) {
            s->v[3 * i + d] += 0.5 * DT * s->f[3 * i + d];
            s->r[3 * i + d] += DT * s->v[3 * i + d];
        }
    }
    forces(s);
    for (int i = 0; i < s->n; i++) {
        if (s->frozen[i]) continue;
        for (int d = 0; d < 3; d++) s->v[3 * i + d] += 0.5 * DT * s->f[3 * i + d];
    }
    rescale(s, T);
}

static void run(sys_t *s, double T, int nsteps)
{
    for (int k = 0; k < nsteps; k++) step(s, T);
}

/* ---------- output ---------- */
static const char *set_name;
static double g_T, g_rho;
static int *g_label;   /* per-atom label */
static double g_z1, g_z2; /* slab interface planes (lower, upper); crystal between */

static double slab_dist(const sys_t *s, double z)
{
    /* signed distance to the nearest interface plane, positive inside
     * [z1, z2) (crystal), periodic in z */
    double Lz = s->L[2];
    double d1 = z - g_z1, d2 = g_z2 - z;
    d1 -= Lz * floor(d1 / Lz + 0.5);
    d2 -= Lz * floor(d2 / Lz + 0.5);
    double zc = z - g_z1;
    zc -= Lz * floor(zc / Lz);
    int inside = zc < (g_z2 - g_z1);
    double m = fmin(fabs(d1), fabs(d2));
    return inside ? m : -m;
}

static void snapshot(sys_t *s, int snap, int is_slab)
{
    int n = s->n;
    double *q4 = malloc(sizeof(double) * n), *q6 = malloc(sizeof(double) * n);
    double *b4 = malloc(sizeof(double) * n), *b6 = malloc(sizeof(double) * n);
    int *cn = malloc(sizeof(int) * n);
    int rc = steinhardt_compute_periodic(n, s->r, s->L, STEINHARDT_RQ_DEFAULT, q4, q6, b4, b6, cn);
    if (rc != 0) {
        fprintf(stderr, "steinhardt_compute_periodic failed: %d\n", rc);
        exit(1);
    }
    double mb6 = 0, fx = 0;
    for (int i = 0; i < n; i++) {
        mb6 += b6[i];
        fx += b6[i] > QBAR6_XTAL;
        double z = s->r[3 * i + 2];
        double dist = is_slab ? slab_dist(s, z) : NAN;
        printf("%s,%.3f,%.3f,%d,%d,%d,%.6f,%.6f,%.10g,%.10g,%.10g,%.10g,%d\n", set_name, g_T,
               g_rho, snap, i, g_label[i], z, dist, b4[i], b6[i], q4[i], q6[i], cn[i]);
    }
    fprintf(stderr, "%-7s T=%.3f rho=%.3f snap=%d  Tinst=%.4f PE/N=%.4f <qbar6>=%.4f frac(qbar6>%.2f)=%.4f\n",
            set_name, g_T, g_rho, snap, temperature(s), s->pe / n, mb6 / n, QBAR6_XTAL, fx / n);
    free(q4); free(q6); free(b4); free(b6); free(cn);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "usage: %s crystal|liquid|slab T rho ncell nsnap seed\n", argv[0]);
        return 1;
    }
    set_name = argv[1];
    double T = atof(argv[2]), rho = atof(argv[3]);
    int nc = atoi(argv[4]), nsnap = atoi(argv[5]);
    rng_s = 0x9E3779B97F4A7C15ULL ^ (unsigned long long)atoll(argv[6]) * 0xBF58476D1CE4E5B9ULL;
    if (!rng_s) rng_s = 1;
    for (int k = 0; k < 10; k++) urand();
    g_T = T;
    g_rho = rho;

    sys_t s;
    int is_slab = !strcmp(set_name, "slab");
    if (is_slab)
        fcc_init(&s, nc, nc, 2 * nc, rho);
    else
        fcc_init(&s, nc, nc, nc, rho);
    g_label = calloc(s.n, sizeof(int));
    forces(&s);

    if (!strcmp(set_name, "crystal")) {
        for (int i = 0; i < s.n; i++) g_label[i] = 1;
        init_velocities(&s, T);
        run(&s, T, EQ_STEPS);
        for (int k = 0; k < nsnap; k++) {
            if (k) run(&s, T, SNAP_EVERY);
            snapshot(&s, k, 0);
        }
    } else if (!strcmp(set_name, "liquid")) {
        init_velocities(&s, T_MELT);
        run(&s, T_MELT, MELT_STEPS);
        rescale(&s, T); /* instant quench */
        run(&s, T, EQ_STEPS);
        for (int k = 0; k < nsnap; k++) {
            if (k) run(&s, T, SNAP_EVERY);
            snapshot(&s, k, 0);
        }
    } else if (is_slab) {
        double Lz = s.L[2];
        /* Lattice (001) layers sit at z = (k/2 + 1/4) a. With nz = 2*ncell
         * cells, Lz/4 and 3Lz/4 are whole multiples of a, i.e. exactly halfway
         * between two layers, so whole layers fall on one side. */
        g_z1 = 0.25 * Lz;
        g_z2 = 0.75 * Lz;
        for (int i = 0; i < s.n; i++) {
            double z0 = s.r[3 * i + 2];
            g_label[i] = (z0 >= g_z1 && z0 < g_z2);
        }
        init_velocities(&s, T);
        run(&s, T, SLAB_THERM_STEPS);
        double *vsave = malloc(sizeof(double) * 3 * s.n);
        memcpy(vsave, s.v, sizeof(double) * 3 * s.n);
        for (int i = 0; i < s.n; i++)
            if (g_label[i]) {
                s.frozen[i] = 1;
                s.v[3 * i] = s.v[3 * i + 1] = s.v[3 * i + 2] = 0;
            }
        init_velocities(&s, T_MELT);
        run(&s, T_MELT, MELT_STEPS);
        rescale(&s, T);
        run(&s, T, SLAB_QUENCH_STEPS);
        /* release: restore the crystal's thermal velocities */
        for (int i = 0; i < s.n; i++)
            if (s.frozen[i]) {
                s.frozen[i] = 0;
                for (int d = 0; d < 3; d++) s.v[3 * i + d] = vsave[3 * i + d];
            }
        rescale(&s, T);
        free(vsave);
        for (int k = 0; k < nsnap; k++) {
            run(&s, T, SLAB_SNAP_EVERY);
            snapshot(&s, k, 1);
        }
    } else {
        fprintf(stderr, "unknown set '%s'\n", set_name);
        return 1;
    }
    return 0;
}
