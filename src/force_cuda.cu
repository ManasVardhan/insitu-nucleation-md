/*----------------------------------------------------------------------
force_cuda.cu: CUDA cell-list Lennard-Jones forces (double precision).

Each call copies resident + cached coordinates to the device, bins them
into the same (lc+2)^3 cell grid as the CPU path (cell sort with CUB),
computes forces with one thread per resident atom over all neighbors
(no Newton's third law), and copies forces and per-atom PE back.
Device buffers persist across calls and grow only when needed.
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "force_cuda.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_), \
          __FILE__,__LINE__); exit(1); } } while (0)
#define NTHREAD 128

/* Device buffers: coordinates (r), cell-sorted coordinates (rs), cell
   keys and atom indices before/after sorting, cell start/end, forces,
   per-atom PE, CUB scratch */
static double *d_r, *d_rs, *d_f, *d_pe;
static int *d_key, *d_keys, *d_val, *d_vals, *d_cstart, *d_cend;
static void *d_tmp;
static size_t cap_tmp;
static int cap_nt, cap_n, cap_cell;

/*--------------------------------------------------------------------*/
__device__ int cell_index(const double *ri, int3 lc2, double3 rc,
                          int *mc) {
/*----------------------------------------------------------------------
Vector & scalar cell index, same rule as the CPU path, clamped to the
grid (cells 0 and lc+1 hold cached atoms).
----------------------------------------------------------------------*/
  mc[0] = (int)((ri[0]+rc.x)/rc.x);
  mc[1] = (int)((ri[1]+rc.y)/rc.y);
  mc[2] = (int)((ri[2]+rc.z)/rc.z);
  mc[0] = min(max(mc[0],0),lc2.x-1);
  mc[1] = min(max(mc[1],0),lc2.y-1);
  mc[2] = min(max(mc[2],0),lc2.z-1);
  return (mc[0]*lc2.y+mc[1])*lc2.z+mc[2];
}

/*--------------------------------------------------------------------*/
__global__ void bin_atoms(int nt, const double *r, int3 lc2, double3 rc,
                          int *key, int *val) {
/*--------------------------------------------------------------------*/
  int i = blockIdx.x*blockDim.x+threadIdx.x, mc[3];
  if (i >= nt) return;
  key[i] = cell_index(&r[3*i],lc2,rc,mc);
  val[i] = i;
}

/*--------------------------------------------------------------------*/
__global__ void cell_bounds(int nt, const int *keys, const int *vals,
                            const double *r, double *rs, int *cstart,
                            int *cend) {
/*----------------------------------------------------------------------
Marks where each cell starts/ends in the sorted order & gathers the
coordinates into sorted order.
----------------------------------------------------------------------*/
  int k = blockIdx.x*blockDim.x+threadIdx.x, c, j;
  if (k >= nt) return;
  c = keys[k];
  if (k == 0    || keys[k-1] != c) cstart[c] = k;
  if (k == nt-1 || keys[k+1] != c) cend[c] = k+1;
  j = vals[k];
  rs[3*k] = r[3*j]; rs[3*k+1] = r[3*j+1]; rs[3*k+2] = r[3*j+2];
}

/*--------------------------------------------------------------------*/
__global__ void lj_force(int n, const double *r, const double *rs,
                         const int *vals, const int *cstart,
                         const int *cend, int3 lc2, double3 rc,
                         double rcut, double Uc, double Duc,
                         double *f, double *pe) {
/*----------------------------------------------------------------------
One thread per resident atom i: sums the force & half the pair energy
over every atom j (resident or cached) within rcut.
----------------------------------------------------------------------*/
  int i = blockIdx.x*blockDim.x+threadIdx.x, mc[3], m0, m1, m2, c1, k;
  double xi,yi,zi,dx,dy,dz,rr,ri2,ri6,r1,fcVal,fx,fy,fz,e,rrCut;
  if (i >= n) return;

  xi = r[3*i]; yi = r[3*i+1]; zi = r[3*i+2];
  cell_index(&r[3*i],lc2,rc,mc);
  rrCut = rcut*rcut;
  fx = fy = fz = e = 0.0;

  for (m0=mc[0]-1; m0<=mc[0]+1; m0++) {
    if (m0 < 0 || m0 >= lc2.x) continue;
    for (m1=mc[1]-1; m1<=mc[1]+1; m1++) {
      if (m1 < 0 || m1 >= lc2.y) continue;
      for (m2=mc[2]-1; m2<=mc[2]+1; m2++) {
        if (m2 < 0 || m2 >= lc2.z) continue;
        c1 = (m0*lc2.y+m1)*lc2.z+m2;
        for (k=cstart[c1]; k<cend[c1]; k++) {
          if (vals[k] == i) continue;
          dx = xi-rs[3*k]; dy = yi-rs[3*k+1]; dz = zi-rs[3*k+2];
          rr = dx*dx+dy*dy+dz*dz;
          if (rr < rrCut) {
            ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
            fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
            e += 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-rcut);
            fx += fcVal*dx; fy += fcVal*dy; fz += fcVal*dz;
          }
        }
      }
    }
  }
  f[3*i] = fx; f[3*i+1] = fy; f[3*i+2] = fz;
  pe[i] = 0.5*e;
}

/*--------------------------------------------------------------------*/
static void grow(void **p, size_t bytes) {
/*--------------------------------------------------------------------*/
  if (*p) CUDA_CHECK(cudaFree(*p));
  CUDA_CHECK(cudaMalloc(p,bytes));
}

/*--------------------------------------------------------------------*/
extern "C" void force_cuda_init(int local_rank, double *r, size_t rbytes,
                                double *f, size_t fbytes, double *pe,
                                size_t pebytes) {
/*--------------------------------------------------------------------*/
  int ndev, dev;
  cudaDeviceProp prop;

  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {fprintf(stderr,"No CUDA device\n"); exit(1);}
  dev = local_rank%ndev;
  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
  printf("local rank %d -> GPU %d (%s)\n",local_rank,dev,prop.name);

  /* Pinned host memory for faster transfers; optional */
  if (cudaHostRegister(r,rbytes,cudaHostRegisterDefault) != cudaSuccess ||
      cudaHostRegister(f,fbytes,cudaHostRegisterDefault) != cudaSuccess ||
      cudaHostRegister(pe,pebytes,cudaHostRegisterDefault) != cudaSuccess) {
    cudaGetLastError();
    printf("local rank %d: cudaHostRegister failed, using pageable memory\n",
           local_rank);
  }
}

/*--------------------------------------------------------------------*/
extern "C" double force_cuda(int n, int nb, const double *r, double *f,
                             double *pe, const int lc[3],
                             const double rc[3], double rcut, double Uc,
                             double Duc) {
/*--------------------------------------------------------------------*/
  int nt = n+nb, ncell, nbit, i, nblk;
  int3 lc2 = make_int3(lc[0]+2,lc[1]+2,lc[2]+2);
  double3 rcv = make_double3(rc[0],rc[1],rc[2]);
  size_t tmp_bytes = 0;
  double lpe = 0.0;

  ncell = lc2.x*lc2.y*lc2.z;
  for (nbit=1; (1<<nbit) < ncell; nbit++);

  /* Grow persistent buffers (with 20% headroom) */
  if (nt > cap_nt) {
    cap_nt = (int)(1.2*nt)+1024;
    grow((void**)&d_r,   sizeof(double)*3*cap_nt);
    grow((void**)&d_rs,  sizeof(double)*3*cap_nt);
    grow((void**)&d_key, sizeof(int)*cap_nt);
    grow((void**)&d_keys,sizeof(int)*cap_nt);
    grow((void**)&d_val, sizeof(int)*cap_nt);
    grow((void**)&d_vals,sizeof(int)*cap_nt);
  }
  if (n > cap_n) {
    cap_n = (int)(1.2*n)+1024;
    grow((void**)&d_f, sizeof(double)*3*cap_n);
    grow((void**)&d_pe,sizeof(double)*cap_n);
  }
  if (ncell > cap_cell) {
    cap_cell = ncell;
    grow((void**)&d_cstart,sizeof(int)*cap_cell);
    grow((void**)&d_cend,  sizeof(int)*cap_cell);
  }
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL,tmp_bytes,d_key,d_keys,
             d_val,d_vals,nt,0,nbit));
  if (tmp_bytes > cap_tmp) {
    cap_tmp = tmp_bytes;
    grow(&d_tmp,cap_tmp);
  }

  /* H->D coordinates, bin & sort by cell */
  CUDA_CHECK(cudaMemcpy(d_r,r,sizeof(double)*3*nt,cudaMemcpyHostToDevice));
  nblk = (nt+NTHREAD-1)/NTHREAD;
  bin_atoms<<<nblk,NTHREAD>>>(nt,d_r,lc2,rcv,d_key,d_val);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_tmp,tmp_bytes,d_key,d_keys,
             d_val,d_vals,nt,0,nbit));
  CUDA_CHECK(cudaMemset(d_cstart,0,sizeof(int)*ncell));
  CUDA_CHECK(cudaMemset(d_cend,  0,sizeof(int)*ncell));
  cell_bounds<<<nblk,NTHREAD>>>(nt,d_keys,d_vals,d_r,d_rs,d_cstart,d_cend);

  /* Forces on residents, D->H */
  if (n > 0) {
    lj_force<<<(n+NTHREAD-1)/NTHREAD,NTHREAD>>>(n,d_r,d_rs,d_vals,d_cstart,
        d_cend,lc2,rcv,rcut,Uc,Duc,d_f,d_pe);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(f,d_f,sizeof(double)*3*n,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pe,d_pe,sizeof(double)*n,cudaMemcpyDeviceToHost));
  }
  for (i=0; i<n; i++) lpe += pe[i];
  return lpe;
}

/*--------------------------------------------------------------------*/
extern "C" void force_cuda_finalize(void) {
/*--------------------------------------------------------------------*/
  cudaFree(d_r); cudaFree(d_rs); cudaFree(d_f); cudaFree(d_pe);
  cudaFree(d_key); cudaFree(d_keys); cudaFree(d_val); cudaFree(d_vals);
  cudaFree(d_cstart); cudaFree(d_cend); cudaFree(d_tmp);
}
