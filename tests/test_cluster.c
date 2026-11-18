/* Unit test for cluster_find(): periodic wrap, chain bonding, cutoff. */
#include <stdio.h>
#include <stdlib.h>

#include "cluster.h"

static int fails = 0;
static void check(int cond, const char *what)
{
    printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond) fails++;
}

int main(void)
{
    const double box[3] = {10.0, 10.0, 10.0};
    cluster_info ci;
    unsigned char in[16];

    /* A chain of 4 atoms across the x boundary (spacing 1.0), a pair, and
       a single atom; plus one atom at 1.5 + 1e-9 from the pair (not bonded). */
    double p[] = {8.5, 5, 5, 9.5, 5, 5, 0.5, 5, 5, 1.5, 5, 5, /* chain */
                  5, 2, 2, 5, 3, 2,                           /* pair */
                  5, 5, 8,                                    /* single */
                  5, 3 + 1.5 + 1e-9, 2};                      /* just outside */
    int n = 8;
    check(cluster_find(n, p, box, 1.5, 2, in, &ci) == 0, "returns 0");
    check(ci.nmax == 4, "largest cluster wraps through the boundary (4)");
    check(ci.n2 == 2, "second largest is the pair (2)");
    check(ci.nclus == 4, "4 clusters in total");
    check(ci.nbig == 2, "2 clusters with >= 2 atoms");
    check(in[0] && in[1] && in[2] && in[3] && !in[4] && !in[7], "membership of the largest");

    /* Unwrapped input gives the same answer */
    p[6] += 10.0; p[9] -= 20.0;
    cluster_find(n, p, box, 1.5, 2, in, &ci);
    check(ci.nmax == 4 && ci.n2 == 2, "images outside the box are wrapped");

    check(cluster_find(0, NULL, box, 1.5, 2, NULL, &ci) == 0 && ci.nmax == 0, "empty input");
    check(cluster_find(n, p, box, 6.0, 2, NULL, &ci) == -1, "rc > box/2 rejected");

    printf("cluster test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
