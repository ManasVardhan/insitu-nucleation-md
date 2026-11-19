/*----------------------------------------------------------------------
pmd.h is an include file for a parallel MD program, pmd.c.
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

/* Constants------------------------------------------------------------

NMAX = Maximum # of atoms per processor
NEMAX = Maximum # of augmented (= resident + copied) atoms
NDBUF = Size of a double-precision buffer, dbuf
      > 8*(# of boundary atoms for each neighbor)
NBMAX = Maximum # of copied boundary atoms per neighbor.
NCLMAX = Maximum # of cells per processor.
RCUT = Potential cut-off length
MOVED_OUT: Signifies a moved-out resident atom in function atom_move.
EMPTY: Signifies the end of a linked list.
----------------------------------------------------------------------*/
/* sized for ~1.1M residents on one rank (one GPU at N ~ 1M) */
#define NMAX 1200000
#define NEMAX 2000000
#define NDBUF 2000000
#define NBMAX 250000
#define NCLMAX 500000
#define RCUT 2.5
#define MOVED_OUT -1.0e10
#define EMPTY -1
/* Constants for the random number generator */
#define D2P31M 2147483647.0
#define DMUL 16807.0

/* Variables------------------------------------------------------------

al[0|1|2] = Box length per processor in the x|y|z direction.
n = # of resident atoms in this processor.
nb = # of copied boundary atoms from neighbors.
nglob = Total # of atoms summed over processors.
r[NEMAX][3]: r[i][0|1|2] is the x|y|z coordinate of atom i (including
  the copied atoms).
rv[NEMAX][3]: rv[i][0|1|2] is the x|y|z velocity of atom i (including
  the copied atoms).
ra[NMAX][3]: ra[i][0|1|2] is the x|y|z acceleration on atom i.
pea[NMAX]: Potential energy of resident atom i (pair energies split
  half and half).
gid[NEMAX]: Global atom ID (0..nglob-1) of atom i, including copies.
ity[NEMAX]: Atom type, 0 = normal, 1 = seed atom.
dbuf[NDBUF]: Buffer for sending double-precision data
dbufr[NDBUF]:           receiving
vproc[0|1|2] = # of processors in the x|y|z direction.
nproc = # of processors = vproc[0]*vproc[1]*vproc[2].
sid = Sequential processor ID.
vid[3] = Vector processor ID;
  sid = vid[0]*vproc[1]*vproc[2] + vid[1]*vproc[2] + vid[2].
NN[6]: NN[ku] is the node ID of the neighbor specified by a neighbor.
  index, ku.  The neighbor index is defined as:
  ku = 0: xlow  (West );
       1: xhigh (East );
       2: ylow  (South);
       3: yhigh (North);
       4: zlow  (Down );
       5: zhigh (Up   ).
sv[6][3]: sv[ku][] is the shift vector to the ku-th neighbor.
myparity[0|1|2] = Parity of vector processor ID in the x|y|z direction.
lsb[6][NBMAX]: lsb[ku][0] is the total # of boundary atoms to be sent
  to neighbor ku; lsb[ku][k] is the atom ID, used in r, of the k-th
  atom to be sent.
status: Returned by MPI message-passing routines.
cpu: Elapsed wall-clock time in seconds.
comt: Communication time in seconds.
fct: Force computation time in seconds.
lc[3]: lc[0|1|2] is the # of cells in the x|y|z direction.
rc[3]: rc[0|1|2] is the length of a cell in the x|y|z direction.
lscl[NEMAX]: Linked cell lists.
head[NCLMAX]: Headers for the linked cell lists.
kinEnergy = Kinetic energy.
potEnergy = Potential energy.
totEnergy = Total energy.
temperature = Current temperature (mobile atoms only).
stepCount = Current time step.
seedFrozen = 1 while the seed atoms are held at their lattice sites.
simcomm = Communicator of the MD (SIM) ranks; MPI_COMM_WORLD unless the
  in situ build splits off analysis ranks.
----------------------------------------------------------------------*/
double al[3];
int n,nb,nglob;
double r[NEMAX][3],rv[NEMAX][3],ra[NMAX][3],pea[NMAX];
int gid[NEMAX],ity[NEMAX];
double dbuf[NDBUF],dbufr[NDBUF];
int vproc[3] = {1,1,2}, nproc = 2;
int sid,vid[3],nn[6],myparity[3];
double sv[6][3];
int lsb[6][NBMAX];
MPI_Status status;
double cpu,comt,fct;
int head[NCLMAX],lscl[NEMAX],lc[3];
double rc[3];
double kinEnergy,potEnergy,totEnergy,temperature;
int stepCount;
int seedFrozen;
MPI_Comm simcomm;
double DeltaTH;    /* Half the time step */
double Uc, Duc;    /* Potential cut-off parameters */

/* Input data-----------------------------------------------------------

Control data: keyword-value file (default pmd.in, or argv[1]).
----------------------------------------------------------------------*/
int InitUcell[3];   /* Number of unit cells per processor */
double Density;     /* Number density of atoms (in reduced unit) */
double InitTemp;    /* Starting temperature (in reduced unit) */
double DeltaT;      /* Size of a time step (in reduced unit) */
int StepLimit;      /* Number of time steps to be simulated */
int StepAvg;        /* Reporting interval for statistical data */

/* Protocol: melt at TMelt for StepMelt steps, ramp linearly to TQuench
   over StepQuench steps, hold for StepHold steps, then optionally run
   StepNVE steps with the thermostat off. */
int Thermostat;     /* 0 = NVE throughout */
double TauT;        /* Berendsen time constant (= DeltaT: plain rescaling) */
double TMelt,TQuench;
int StepMelt,StepQuench,StepHold,StepNVE;
/* Seeding: atoms within RSeed of the box center stay frozen at their
   FCC sites until StepMelt+StepQuench+StepRelease */
double RSeed;
int StepRelease,StepFreeze;
int XYZInterval;    /* XYZ output interval, 0 = off */
char XYZPrefix[256];
int ForceCheck;     /* CUDA build: GPU vs CPU force check interval */
double RandSeed;
/* In situ analysis (-DINSITU build, insitu.c): InsituRanks ANA ranks in
   addition to the vproc SIM ranks (0 = analysis off); analysis every
   InsituInterval steps from step InsituStart; movie snapshot every
   InsituMovie analysis steps (0 = off); InsituCheck = 1 writes full
   dumps for the post hoc equivalence check. */
int InsituRanks,InsituInterval,InsituStart,InsituMovie,InsituCheck;
int InsituClusMin;
double InsituThr;
char InsituPrefix[256];

/* Functions & function prototypes------------------------------------*/

double SignR(double v,double x) {if (x > 0) return v; else return -v;}
double Dmod(double a, double b) {
  int n;
  n = (int) (a/b);
  return (a - b*n);
}
double RandR(double *seed) {
  *seed = Dmod(*seed*DMUL,D2P31M);
  return (*seed/D2P31M);
}
void RandVec3(double *p, double *seed) {
  double x,y,s = 2.0;
  while (s > 1.0) {
    x = 2.0*RandR(seed) - 1.0; y = 2.0*RandR(seed) - 1.0; s = x*x + y*y;
  }
  p[2] = 1.0 - 2.0*s; s = 2.0*sqrt(1.0 - s); p[0] = s*x; p[1] = s*y;
}

void init_params(char *fname);
void set_topology();
void init_conf();
void single_step();
void half_kick();
void atom_copy();
int halo_copy(double w, double (*x)[3], int *g, int nmax);
void insitu_send();
void compute_accel();
double compute_accel_cpu(double (*f)[3], double *pe);
void thermostat();
void zero_momentum();
double target_temp(int step);
void eval_props();
void atom_move();
void write_xyz();
void force_check();
void abort_if(int cond, const char *msg);
void check_ids();
int bbd(double* ri, int ku, double w);
int bmv(double* ri, int ku);
/*--------------------------------------------------------------------*/
