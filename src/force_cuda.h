/*----------------------------------------------------------------------
force_cuda.h: interface to the CUDA Lennard-Jones force path.
----------------------------------------------------------------------*/
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Selects GPU (local_rank % # of devices) and pins the host arrays */
void force_cuda_init(int local_rank, double *r, size_t rbytes,
                     double *f, size_t fbytes, double *pe, size_t pebytes);

/* Forces f[0:n-1][3] and per-atom PE pe[0:n-1] on the n resident atoms
   of r[0:n+nb-1][3]; returns the local PE sum */
double force_cuda(int n, int nb, const double *r, double *f, double *pe,
                  const int lc[3], const double rc[3], double rcut,
                  double Uc, double Duc);

void force_cuda_finalize(void);

#ifdef __cplusplus
}
#endif
