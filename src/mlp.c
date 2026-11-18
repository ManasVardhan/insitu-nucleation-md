/*
 * mlp.c: hand-written forward pass, 5 -> 16 -> 16 -> 2 with ReLU.
 * See mlp.h. Weights come from the generated header mlp_weights.h.
 */
#include "mlp.h"
#include "mlp_weights.h"

#include <math.h>
#include <stddef.h>

#if MLP_NIN != MLP_NFEAT || MLP_NOUT != 2
#error "mlp_weights.h does not match mlp.h"
#endif

int mlp_forward(const double feat[MLP_NFEAT], double logits[2])
{
    double x[MLP_NIN], h1[MLP_NH1], h2[MLP_NH2];
    for (int i = 0; i < MLP_NIN; i++) x[i] = (feat[i] - MLP_MEAN[i]) / MLP_STD[i];
    for (int o = 0; o < MLP_NH1; o++) {
        double s = MLP_B1[o];
        for (int i = 0; i < MLP_NIN; i++) s += MLP_W1[o][i] * x[i];
        h1[o] = s > 0.0 ? s : 0.0;
    }
    for (int o = 0; o < MLP_NH2; o++) {
        double s = MLP_B2[o];
        for (int i = 0; i < MLP_NH1; i++) s += MLP_W2[o][i] * h1[i];
        h2[o] = s > 0.0 ? s : 0.0;
    }
    for (int o = 0; o < MLP_NOUT; o++) {
        double s = MLP_B3[o];
        for (int i = 0; i < MLP_NH2; i++) s += MLP_W3[o][i] * h2[i];
        logits[o] = s;
    }
    return logits[1] > logits[0] ? MLP_LABEL_CRYSTAL : MLP_LABEL_LIQUID;
}

int mlp_classify(const double feat[MLP_NFEAT], double *p_crystal)
{
    double l[2];
    int label = mlp_forward(feat, l);
    /* softmax(l)[1] = 1 / (1 + exp(l0 - l1)), stable for large |l0 - l1| */
    if (p_crystal) {
        double d = l[0] - l[1];
        *p_crystal = d > 0 ? exp(-d) / (1.0 + exp(-d)) : 1.0 / (1.0 + exp(d));
    }
    return label;
}

void mlp_classify_batch(int n, const double *feat, int *label, double *p_crystal)
{
    for (int i = 0; i < n; i++) {
        double p;
        int lab = mlp_classify(&feat[(size_t)i * MLP_NFEAT], &p);
        if (label) label[i] = lab;
        if (p_crystal) p_crystal[i] = p;
    }
}
