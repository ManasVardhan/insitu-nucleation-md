/*
 * posthoc.c: post hoc equivalence check of the in situ analysis (T2.5).
 *
 * For every analysis step listed in <prefix>_summary.dat, reads the
 * full-precision dump <prefix>_full_<step>.bin written by the SIM ranks,
 * recomputes features with steinhardt_compute_periodic() on the whole
 * periodic box, labels with the MLP and the qbar6 threshold, finds the
 * largest clusters, and compares atom by atom (by global ID) with
 *   - the in situ label file <prefix>_labels.bin (labels and largest-
 *     cluster membership of every crystalline atom),
 *   - the in situ largest-cluster sizes in the summary, and
 *   - if present, the in situ per-atom feature dumps
 *     <prefix>_feat_<step>_a<k>.bin (feature and probability differences).
 *
 * Usage: posthoc <prefix> [thr] [clusmin]     (defaults 0.308, 20)
 * Exit status 0 if all labels and cluster sizes agree, 1 otherwise.
 * Plain C99, no MPI.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cluster.h"
#include "mlp.h"
#include "steinhardt.h"

#define BIT_MLP 1
#define BIT_THR 2
#define BIT_MAXMLP 4
#define BIT_MAXTHR 8

static void *xmalloc(size_t n)
{
    void *p = malloc(n > 0 ? n : 1);
    if (!p) {
        fprintf(stderr, "posthoc: out of memory\n");
        exit(2);
    }
    return p;
}

static int read_exact(FILE *fp, void *buf, size_t sz, size_t n)
{
    return fread(buf, sz, n, fp) == n;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <prefix> [thr] [clusmin]\n", argv[0]);
        return 2;
    }
    const char *prefix = argv[1];
    double thr = argc > 2 ? atof(argv[2]) : 0.308;
    int clusmin = argc > 3 ? atoi(argv[3]) : 20;
    const double rq = STEINHARDT_RQ_DEFAULT;
    char fname[1024], line[1024];

    snprintf(fname, sizeof(fname), "%s_summary.dat", prefix);
    FILE *fsum = fopen(fname, "r");
    snprintf(fname, sizeof(fname), "%s_labels.bin", prefix);
    FILE *flab = fopen(fname, "rb");
    if (!fsum || !flab) {
        fprintf(stderr, "posthoc: cannot open %s_summary.dat / _labels.bin\n", prefix);
        return 2;
    }

    long tot_steps = 0, tot_atoms = 0, tot_mis_mlp = 0, tot_mis_thr = 0;
    long tot_mis_max = 0, tot_bad_nmax = 0, tot_feat_atoms = 0, tot_coord_mis = 0;
    double gmax_df[5] = {0, 0, 0, 0, 0}, gmax_dp = 0.0;
    int have_feat_all = 1;

    printf("# step nglob ncry_mlp ncry_thr mis_mlp mis_thr mis_maxbits "
           "nmax_mlp(insitu/post) nmax_thr(insitu/post) feat_atoms "
           "max|dqbar6| max|dpcr|\n");
    while (fgets(line, sizeof(line), fsum)) {
        if (line[0] == '#') continue;
        int step, ncm, nct, nmm, nmt;
        double tm;
        if (sscanf(line, "%d %lf %d %d %d %d", &step, &tm, &ncm, &nct, &nmm, &nmt) != 6) continue;

        /* Full-precision dump */
        snprintf(fname, sizeof(fname), "%s_full_%08d.bin", prefix, step);
        FILE *fp = fopen(fname, "rb");
        if (!fp) {
            fprintf(stderr, "posthoc: missing %s\n", fname);
            return 2;
        }
        int hdr[2];
        double box[3];
        if (!read_exact(fp, hdr, sizeof(int), 2) || !read_exact(fp, box, sizeof(double), 3)) return 2;
        int n = hdr[0];
        if (hdr[1] != step) {
            fprintf(stderr, "posthoc: step mismatch in %s\n", fname);
            return 2;
        }
        double *pos = xmalloc(sizeof(double) * 3 * (size_t)n);
        if (!read_exact(fp, pos, sizeof(double), 3 * (size_t)n)) return 2;
        fclose(fp);

        /* Post hoc features and labels (atom index = global ID) */
        double *q4 = xmalloc(sizeof(double) * n), *q6 = xmalloc(sizeof(double) * n);
        double *b4 = xmalloc(sizeof(double) * n), *b6 = xmalloc(sizeof(double) * n);
        int *coord = xmalloc(sizeof(int) * n), *lab = xmalloc(sizeof(int) * n);
        double *feat = xmalloc(sizeof(double) * 5 * (size_t)n), *pcr = xmalloc(sizeof(double) * n);
        if (steinhardt_compute_periodic(n, pos, box, rq, q4, q6, b4, b6, coord) != 0) return 2;
        for (int i = 0; i < n; i++)
            steinhardt_features(q4[i], q6[i], b4[i], b6[i], coord[i], feat + 5 * (size_t)i);
        mlp_classify_batch(n, feat, lab, pcr);
        unsigned char *bits = xmalloc(n);
        for (int i = 0; i < n; i++)
            bits[i] = (lab[i] == MLP_LABEL_CRYSTAL ? BIT_MLP : 0) | (b6[i] > thr ? BIT_THR : 0);

        /* Post hoc clusters, atoms in ID order (as on the ANA root) */
        double *cp = xmalloc(sizeof(double) * 3 * (size_t)n);
        unsigned char *inmax = xmalloc(n);
        cluster_info ci[2];
        for (int a = 0; a < 2; a++) {
            int bit = a == 0 ? BIT_MLP : BIT_THR, m = 0;
            for (int i = 0; i < n; i++)
                if (bits[i] & bit) {
                    memcpy(cp + 3 * m, pos + 3 * (size_t)i, 3 * sizeof(double));
                    m++;
                }
            if (cluster_find(m, cp, box, rq, clusmin, inmax, &ci[a]) != 0) return 2;
            for (int i = 0, k = 0; i < n; i++)
                if (bits[i] & bit) {
                    if (inmax[k]) bits[i] |= (a == 0 ? BIT_MAXMLP : BIT_MAXTHR);
                    k++;
                }
        }

        /* In situ labels for this step */
        int lh[2];
        if (!read_exact(flab, lh, sizeof(int), 2) || lh[0] != step) {
            fprintf(stderr, "posthoc: label file out of sync at step %d\n", step);
            return 2;
        }
        int *ids = xmalloc(sizeof(int) * (size_t)(lh[1] > 0 ? lh[1] : 1));
        unsigned char *ib = xmalloc(lh[1] > 0 ? lh[1] : 1);
        if (!read_exact(flab, ids, sizeof(int), lh[1]) || !read_exact(flab, ib, 1, lh[1])) return 2;
        unsigned char *isb = xmalloc(n);
        memset(isb, 0, n);
        for (int k = 0; k < lh[1]; k++) isb[ids[k]] = ib[k];

        long mis_mlp = 0, mis_thr = 0, mis_max = 0, pc_mlp = 0, pc_thr = 0;
        for (int i = 0; i < n; i++) {
            if (bits[i] & BIT_MLP) pc_mlp++;
            if (bits[i] & BIT_THR) pc_thr++;
            int d = bits[i] ^ isb[i];
            if (d & BIT_MLP) {
                mis_mlp++;
                printf("  MISMATCH mlp step %d id %d insitu %d posthoc %d p_post %.17g\n", step, i,
                       isb[i] & 1, bits[i] & 1, pcr[i]);
            }
            if (d & BIT_THR) {
                mis_thr++;
                printf("  MISMATCH thr step %d id %d insitu %d posthoc %d qbar6_post %.17g\n", step,
                       i, (isb[i] >> 1) & 1, (bits[i] >> 1) & 1, b6[i]);
            }
            if (d & (BIT_MAXMLP | BIT_MAXTHR)) mis_max++;
        }

        /* In situ per-atom features, if dumped */
        long nfa = 0;
        double max_df[5] = {0, 0, 0, 0, 0}, max_dp = 0.0;
        unsigned char *seen = xmalloc(n);
        memset(seen, 0, n);
        for (int k = 0;; k++) {
            snprintf(fname, sizeof(fname), "%s_feat_%08d_a%d.bin", prefix, step, k);
            FILE *ff = fopen(fname, "rb");
            if (!ff) break;
            int c;
            if (!read_exact(ff, &c, sizeof(int), 1)) return 2;
            int *fid = xmalloc(sizeof(int) * (size_t)(c > 0 ? c : 1));
            double *ffeat = xmalloc(sizeof(double) * 5 * (size_t)(c > 0 ? c : 1));
            double *fp2 = xmalloc(sizeof(double) * (size_t)(c > 0 ? c : 1));
            unsigned char *fb = xmalloc(c > 0 ? c : 1);
            if (!read_exact(ff, fid, sizeof(int), c) || !read_exact(ff, ffeat, sizeof(double), 5 * (size_t)c) ||
                !read_exact(ff, fp2, sizeof(double), c) || !read_exact(ff, fb, 1, c))
                return 2;
            fclose(ff);
            for (int j = 0; j < c; j++) {
                int id = fid[j];
                if (id < 0 || id >= n || seen[id]) {
                    printf("  FEATURE DUMP: id %d out of range or duplicated\n", id);
                    continue;
                }
                seen[id] = 1;
                nfa++;
                for (int f = 0; f < 5; f++) {
                    double d = fabs(ffeat[5 * (size_t)j + f] - feat[5 * (size_t)id + f]);
                    if (d > max_df[f]) max_df[f] = d;
                }
                if (ffeat[5 * (size_t)j + 4] != feat[5 * (size_t)id + 4]) tot_coord_mis++;
                double d = fabs(fp2[j] - pcr[id]);
                if (d > max_dp) max_dp = d;
            }
            free(fid);
            free(ffeat);
            free(fp2);
            free(fb);
        }
        if (nfa != n) have_feat_all = 0;
        for (int f = 0; f < 5; f++)
            if (max_df[f] > gmax_df[f]) gmax_df[f] = max_df[f];
        if (max_dp > gmax_dp) gmax_dp = max_dp;

        int bad_nmax = (nmm != ci[0].nmax) || (nmt != ci[1].nmax) || (ncm != pc_mlp) || (nct != pc_thr);
        printf("%d %d %ld %ld %ld %ld %ld %d/%d %d/%d %ld %.3e %.3e%s\n", step, n, pc_mlp, pc_thr, mis_mlp,
               mis_thr, mis_max, nmm, ci[0].nmax, nmt, ci[1].nmax, nfa, max_df[1], max_dp,
               bad_nmax ? "  NMAX/COUNT MISMATCH" : "");

        tot_steps++;
        tot_atoms += n;
        tot_mis_mlp += mis_mlp;
        tot_mis_thr += mis_thr;
        tot_mis_max += mis_max;
        tot_bad_nmax += bad_nmax;
        tot_feat_atoms += nfa;

        free(pos); free(q4); free(q6); free(b4); free(b6); free(coord); free(lab);
        free(feat); free(pcr); free(bits); free(cp); free(inmax); free(ids); free(ib);
        free(isb); free(seen);
    }
    fclose(fsum);
    fclose(flab);

    printf("POSTHOC steps %ld atom_labels %ld mlp_mismatch %ld thr_mismatch %ld "
           "maxcluster_membership_mismatch %ld nmax_or_count_mismatch_steps %ld\n",
           tot_steps, tot_atoms, tot_mis_mlp, tot_mis_thr, tot_mis_max, tot_bad_nmax);
    printf("POSTHOC features compared %ld (all atoms every step: %s) coord_mismatch %ld "
           "max|d| qbar4 %.3e qbar6 %.3e q4 %.3e q6 %.3e p_crystal %.3e\n",
           tot_feat_atoms, have_feat_all ? "yes" : "no", tot_coord_mis, gmax_df[0], gmax_df[1],
           gmax_df[2], gmax_df[3], gmax_dp);
    int ok = tot_steps > 0 && tot_mis_mlp == 0 && tot_mis_thr == 0 && tot_mis_max == 0 && tot_bad_nmax == 0;
    printf("POSTHOC %s\n", ok ? "EXACT AGREEMENT" : "DIFFERENCES FOUND");
    return ok ? 0 : 1;
}
