/*
 * test_mlp: C forward pass vs PyTorch on the samples in data/mlp_ref.csv
 * (written by train/export_weights.py).
 *   pass criterion: max |logit_C - logit_torch64| < 1e-6 and identical labels
 *   info:           max |logit_C - logit_torch32| (float32 model)
 */
#include "mlp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "data/mlp_ref.csv";
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    char line[1024];
    if (!fgets(line, sizeof line, fp)) return 2; /* header */
    int n = 0, label_mismatch = 0, n_xtal = 0;
    double e64 = 0, e32 = 0, ep = 0;
    double f[MLP_NFEAT], r64[2], r32[2], p64;
    while (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &f[0], &f[1], &f[2], &f[3],
                  &f[4], &r64[0], &r64[1], &r32[0], &r32[1], &p64) == 10) {
        double l[2], p;
        int lab = mlp_forward(f, l);
        int lab2 = mlp_classify(f, &p);
        for (int k = 0; k < 2; k++) {
            e64 = fmax(e64, fabs(l[k] - r64[k]));
            e32 = fmax(e32, fabs(l[k] - r32[k]));
        }
        ep = fmax(ep, fabs(p - p64));
        int ref_lab = r64[1] > r64[0];
        if (lab != ref_lab || lab2 != lab) label_mismatch++;
        n_xtal += lab;
        n++;
    }
    fclose(fp);
    printf("[mlp] samples = %d (%d crystal)\n", n, n_xtal);
    printf("  max |logit C - torch float64|   = %.3e\n", e64);
    printf("  max |p_crystal C - torch64|     = %.3e\n", ep);
    printf("  max |logit C - torch float32|   = %.3e (info)\n", e32);
    printf("  label mismatches                = %d\n", label_mismatch);
    int ok = n >= 1000 && e64 < 1e-6 && ep < 1e-6 && label_mismatch == 0;
    printf("mlp test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
