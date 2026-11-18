/*----------------------------------------------------------------------
Program pmd.c performs parallel molecular-dynamics for Lennard-Jones 
systems using the Message Passing Interface (MPI) standard.
----------------------------------------------------------------------*/
#include "pmd.h"
#ifdef USE_CUDA
#include "force_cuda.h"
#endif

/*--------------------------------------------------------------------*/
int main(int argc, char **argv) {
/*--------------------------------------------------------------------*/
  double cpu1;
  int size;
#ifdef USE_CUDA
  MPI_Comm shm;
  int lrank;
#endif

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Comm_rank(MPI_COMM_WORLD, &sid);  /* My processor ID */
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  setvbuf(stdout,NULL,_IOLBF,0); /* Line-buffered log for batch jobs */

  init_params(argc > 1 ? argv[1] : "pmd.in");
  abort_if(size != nproc, "MPI size does not match vproc");
  /* Vector index of this processor */
  vid[0] = sid/(vproc[1]*vproc[2]);
  vid[1] = (sid/vproc[2])%vproc[1];
  vid[2] = sid%vproc[2];
#ifdef USE_CUDA
  /* One GPU per rank, chosen by the rank index within the node */
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,sid,
                      MPI_INFO_NULL,&shm);
  MPI_Comm_rank(shm,&lrank);
  force_cuda_init(lrank,&r[0][0],sizeof(r),&ra[0][0],sizeof(ra),
                  pea,sizeof(pea));
#endif

  set_topology(); 
  init_conf();
  atom_copy();
  compute_accel(); /* Computes initial accelerations */ 
  stepCount = 0;
  eval_props();
  if (XYZInterval > 0) write_xyz();

  cpu1 = MPI_Wtime();
  comt = fct = 0.0;
  for (stepCount=1; stepCount<=StepLimit; stepCount++) {
    single_step(); 
    if (stepCount%StepAvg == 0) eval_props();
    if (XYZInterval > 0 && stepCount%XYZInterval == 0) write_xyz();
#ifdef USE_CUDA
    if (ForceCheck > 0 && stepCount%ForceCheck == 0) force_check();
#endif
#ifdef CHECK_IDS
    if (stepCount%StepAvg == 0) check_ids();
#endif
  }
  cpu = MPI_Wtime() - cpu1;
  if (sid == 0) {
    printf("CPU & COMT = %le %le\n",cpu,comt);
    printf("TIMING nglob %d nproc %d steps %d wall %.4f step_ms %.4f "
           "force_ms %.4f comm_ms %.4f\n",nglob,nproc,StepLimit,cpu,
           1e3*cpu/StepLimit,1e3*fct/StepLimit,1e3*comt/StepLimit);
  }
#ifdef USE_CUDA
  force_cuda_finalize();
#endif

  MPI_Finalize(); /* Clean up the MPI environment */
  return 0;
}

/*--------------------------------------------------------------------*/
void init_params(char *fname) {
/*----------------------------------------------------------------------
Initializes parameters.  The control file has one "keyword value(s)"
pair per line; '#' starts a comment.
----------------------------------------------------------------------*/
  int a;
  double rr,ri2,ri6,r1;
  char line[512],key[64],*p;
  FILE *fp;

  /* Defaults */
  InitUcell[0] = InitUcell[1] = InitUcell[2] = 8;
  Density = 0.8; InitTemp = 1.0; DeltaT = 0.005;
  StepLimit = 100; StepAvg = 10;
  Thermostat = 0; TauT = 0.1; TMelt = TQuench = 1.0;
  StepMelt = StepQuench = StepHold = StepNVE = 0;
  RSeed = 0.0; StepRelease = 0;
  XYZInterval = 0; strcpy(XYZPrefix,"traj");
  ForceCheck = 0; RandSeed = 13597.0;

  /* Read control parameters */
  fp = fopen(fname,"r");
  abort_if(fp == NULL, "cannot open input file");
  while (fgets(line,sizeof(line),fp)) {
    if ((p = strchr(line,'#')) != NULL) *p = '\0';
    if (sscanf(line,"%63s",key) != 1) continue;
    p = strstr(line,key) + strlen(key);
    if      (!strcmp(key,"vproc"))
      sscanf(p,"%d%d%d",&vproc[0],&vproc[1],&vproc[2]);
    else if (!strcmp(key,"InitUcell"))
      sscanf(p,"%d%d%d",&InitUcell[0],&InitUcell[1],&InitUcell[2]);
    else if (!strcmp(key,"Density"))  sscanf(p,"%le",&Density);
    else if (!strcmp(key,"InitTemp")) sscanf(p,"%le",&InitTemp);
    else if (!strcmp(key,"DeltaT"))   sscanf(p,"%le",&DeltaT);
    else if (!strcmp(key,"StepLimit"))sscanf(p,"%d",&StepLimit);
    else if (!strcmp(key,"StepAvg"))  sscanf(p,"%d",&StepAvg);
    else if (!strcmp(key,"Thermostat")) sscanf(p,"%d",&Thermostat);
    else if (!strcmp(key,"TauT"))     sscanf(p,"%le",&TauT);
    else if (!strcmp(key,"TMelt"))    sscanf(p,"%le",&TMelt);
    else if (!strcmp(key,"TQuench"))  sscanf(p,"%le",&TQuench);
    else if (!strcmp(key,"StepMelt")) sscanf(p,"%d",&StepMelt);
    else if (!strcmp(key,"StepQuench")) sscanf(p,"%d",&StepQuench);
    else if (!strcmp(key,"StepHold")) sscanf(p,"%d",&StepHold);
    else if (!strcmp(key,"StepNVE"))  sscanf(p,"%d",&StepNVE);
    else if (!strcmp(key,"RSeed"))    sscanf(p,"%le",&RSeed);
    else if (!strcmp(key,"StepRelease")) sscanf(p,"%d",&StepRelease);
    else if (!strcmp(key,"XYZInterval")) sscanf(p,"%d",&XYZInterval);
    else if (!strcmp(key,"XYZPrefix")) sscanf(p,"%255s",XYZPrefix);
    else if (!strcmp(key,"ForceCheck")) sscanf(p,"%d",&ForceCheck);
    else if (!strcmp(key,"RandSeed")) sscanf(p,"%le",&RandSeed);
    else if (sid == 0) printf("Unknown keyword: %s\n",key);
  }
  fclose(fp);
  nproc = vproc[0]*vproc[1]*vproc[2];

  /* A protocol, if given, overrides StepLimit */
  if (StepMelt+StepQuench+StepHold+StepNVE > 0)
    StepLimit = StepMelt+StepQuench+StepHold+StepNVE;
  StepFreeze = StepMelt+StepQuench+StepRelease;
  if (sid == 0) {
    printf("Protocol: thermostat %d tau %g | melt %g x %d | ramp %d | "
           "hold %g x %d | nve %d | total %d\n",Thermostat,TauT,TMelt,
           StepMelt,StepQuench,TQuench,StepHold,StepNVE,StepLimit);
    if (RSeed > 0.0) printf("Seed: R %g frozen through step %d\n",
                            RSeed,StepFreeze);
  }

  /* Compute basic parameters */
  DeltaTH = 0.5*DeltaT;
  for (a=0; a<3; a++) al[a] = InitUcell[a]/pow(Density/4.0,1.0/3.0);
  if (sid == 0) printf("al = %e %e %e\n",al[0],al[1],al[2]);

  /* Compute the # of cells for linked cell lists */
  for (a=0; a<3; a++) {
    lc[a] = al[a]/RCUT; 
    rc[a] = al[a]/lc[a];
  }
  if (sid == 0) {
    printf("lc = %d %d %d\n",lc[0],lc[1],lc[2]);
    printf("rc = %e %e %e\n",rc[0],rc[1],rc[2]);
  }

  /* Constants for potential truncation */
  rr = RCUT*RCUT; ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1=sqrt(rr);
  Uc = 4.0*ri6*(ri6 - 1.0);
  Duc = -48.0*ri6*(ri6 - 0.5)/r1;
}

/*--------------------------------------------------------------------*/
void set_topology() {
/*----------------------------------------------------------------------
Defines a logical network topology.  Prepares a neighbor-node ID table, 
nn, & a shift-vector table, sv, for internode message passing.  Also 
prepares the node parity table, myparity.
----------------------------------------------------------------------*/
  /* Integer vectors to specify the six neighbor nodes */
  int iv[6][3] = {
    {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
  };
  int ku,a,k1[3];

  /* Set up neighbor tables, nn & sv */
  for (ku=0; ku<6; ku++) {
    /* Vector index of neighbor ku */
    for (a=0; a<3; a++)
      k1[a] = (vid[a]+iv[ku][a]+vproc[a])%vproc[a];
    /* Scalar neighbor ID, nn */
    nn[ku] = k1[0]*vproc[1]*vproc[2]+k1[1]*vproc[2]+k1[2];
    /* Shift vector, sv */
    for (a=0; a<3; a++) sv[ku][a] = al[a]*iv[ku][a];
  }

  /* Set up the node parity table, myparity */
  for (a=0; a<3; a++) {
    if (vproc[a] == 1) 
      myparity[a] = 2;
    else if (vid[a]%2 == 0)
      myparity[a] = 0;
    else
      myparity[a] = 1;
  }
}

/*--------------------------------------------------------------------*/
void init_conf() {
/*----------------------------------------------------------------------
r are initialized to face-centered cubic (fcc) lattice positions.  
rv are initialized with a random velocity corresponding to Temperature.  
----------------------------------------------------------------------*/
  double c[3],gap[3],e[3],vSum[4],gvSum[4],vMag,rr,d;
  int j,a,nX,nY,nZ,ioff,nseed,gseed;
  double seed;
  /* FCC atoms in the original unit cell */
  double origAtom[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5},
                           {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}; 

  /* Set up a face-centered cubic (fcc) lattice */
  for (a=0; a<3; a++) gap[a] = al[a]/InitUcell[a];
  n = 0;
  for (nZ=0; nZ<InitUcell[2]; nZ++) {
    c[2] = nZ*gap[2];
    for (nY=0; nY<InitUcell[1]; nY++) {
      c[1] = nY*gap[1];
      for (nX=0; nX<InitUcell[0]; nX++) {
        c[0] = nX*gap[0];
        for (j=0; j<4; j++) {
          for (a=0; a<3; a++)
            r[n][a] = c[a] + gap[a]*origAtom[j][a];
          ++n;
        }
      }
    }
  }
  /* Total # of atoms summed over processors */
  MPI_Allreduce(&n,&nglob,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (sid == 0) printf("nglob = %d\n",nglob);

  /* Global atom IDs: ranks own consecutive ID blocks */
  ioff = 0;
  MPI_Exscan(&n,&ioff,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (sid == 0) ioff = 0;
  for (j=0; j<n; j++) {gid[j] = ioff+j; ity[j] = 0;}

  /* Seed atoms: within RSeed of the global box center */
  nseed = 0;
  if (RSeed > 0.0) {
    for (j=0; j<n; j++) {
      for (rr=0.0, a=0; a<3; a++) {
        d = r[j][a]+vid[a]*al[a]-0.5*vproc[a]*al[a];
        rr += d*d;
      }
      if (rr < RSeed*RSeed) {ity[j] = 1; nseed++;}
    }
    MPI_Allreduce(&nseed,&gseed,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if (sid == 0) printf("nseed = %d\n",gseed);
  }
  seedFrozen = (RSeed > 0.0 && StepFreeze >= 0);

  /* Generate random velocities (frozen seed atoms stay at rest) */
  seed = RandSeed+sid;
  vMag = sqrt(3*InitTemp);
  for(a=0; a<4; a++) vSum[a] = 0.0;
  for(j=0; j<n; j++) {
    RandVec3(e,&seed);
    for (a=0; a<3; a++) rv[j][a] = 0.0;
    if (seedFrozen && ity[j]) continue;
    for (a=0; a<3; a++) {
      rv[j][a] = vMag*e[a];
      vSum[a] = vSum[a] + rv[j][a];
    }
    vSum[3] += 1.0;
  }
  MPI_Allreduce(vSum,gvSum,4,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  /* Make the total momentum zero */
  for (a=0; a<3; a++) gvSum[a] /= gvSum[3];
  for (j=0; j<n; j++) {
    if (seedFrozen && ity[j]) continue;
    for(a=0; a<3; a++) rv[j][a] -= gvSum[a];
  }
}

/*--------------------------------------------------------------------*/
void single_step() {
/*----------------------------------------------------------------------
r & rv are propagated by DeltaT using the velocity-Verlet scheme.
----------------------------------------------------------------------*/
  int i,a;

  seedFrozen = (RSeed > 0.0 && stepCount <= StepFreeze);
  half_kick(); /* First half kick to obtain v(t+Dt/2) */
  for (i=0; i<n; i++) { /* Update atomic coordinates to r(t+Dt) */
    if (seedFrozen && ity[i]) continue;
    for (a=0; a<3; a++) r[i][a] = r[i][a] + DeltaT*rv[i][a];
  }
  atom_move();
  atom_copy();
  compute_accel(); /* Computes new accelerations, a(t+Dt) */
  half_kick(); /* Second half kick to obtain v(t+Dt) */
  if (Thermostat && stepCount <= StepMelt+StepQuench+StepHold)
    thermostat();
  if (RSeed > 0.0 && stepCount == StepFreeze+1) zero_momentum();
}

/*--------------------------------------------------------------------*/
void zero_momentum() {
/*----------------------------------------------------------------------
Removes the center-of-mass velocity (used when the seed is released).
----------------------------------------------------------------------*/
  double loc[3],glb[3];
  int i,a;
  for (a=0; a<3; a++) loc[a] = 0.0;
  for (i=0; i<n; i++) for (a=0; a<3; a++) loc[a] += rv[i][a];
  MPI_Allreduce(loc,glb,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  for (i=0; i<n; i++) for (a=0; a<3; a++) rv[i][a] -= glb[a]/nglob;
}

/*--------------------------------------------------------------------*/
double target_temp(int step) {
/*----------------------------------------------------------------------
Target temperature of the melt -> linear quench -> hold protocol.
----------------------------------------------------------------------*/
  if (step <= StepMelt) return TMelt;
  if (step <= StepMelt+StepQuench)
    return TMelt+(TQuench-TMelt)*(step-StepMelt)/(double)StepQuench;
  return TQuench;
}

/*--------------------------------------------------------------------*/
void thermostat() {
/*----------------------------------------------------------------------
Berendsen velocity rescaling toward target_temp(stepCount), using the
global kinetic energy of the mobile atoms.  TauT = DeltaT gives plain
velocity rescaling.
----------------------------------------------------------------------*/
  double loc[2],glb[2],vv,t,lam;
  int i,a;

  loc[0] = loc[1] = 0.0;
  for (i=0; i<n; i++) {
    if (seedFrozen && ity[i]) continue;
    for (vv=0.0, a=0; a<3; a++) vv += rv[i][a]*rv[i][a];
    loc[0] += 0.5*vv; loc[1] += 1.0;
  }
  MPI_Allreduce(loc,glb,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  t = 2.0*glb[0]/(3.0*glb[1]);
  if (t <= 0.0) return;
  lam = sqrt(1.0+DeltaT/TauT*(target_temp(stepCount)/t-1.0));
  for (i=0; i<n; i++) {
    if (seedFrozen && ity[i]) continue;
    for (a=0; a<3; a++) rv[i][a] *= lam;
  }
}

/*--------------------------------------------------------------------*/
void half_kick() {
/*----------------------------------------------------------------------
Accelerates atomic velocities, rv, by half the time step.
----------------------------------------------------------------------*/
  int i,a;
  for (i=0; i<n; i++) {
    if (seedFrozen && ity[i]) continue;
    for (a=0; a<3; a++) rv[i][a] = rv[i][a]+DeltaTH*ra[i][a];
  }
}

/*--------------------------------------------------------------------*/
void atom_copy() {
/*----------------------------------------------------------------------
Exchanges boundary-atom coordinates among neighbor nodes:  Makes 
boundary-atom list, LSB, then sends & receives boundary atoms.
----------------------------------------------------------------------*/
  int kd,kdd,i,ku,inode,nsd,nrc,a;
  int nbnew = 0; /* # of "received" boundary atoms */
  double com1;

/* Main loop over x, y & z directions starts--------------------------*/

  for (kd=0; kd<3; kd++) {

    /* Make a boundary-atom list, LSB---------------------------------*/

    /* Reset the # of to-be-copied atoms for lower&higher directions */
    for (kdd=0; kdd<2; kdd++) lsb[2*kd+kdd][0] = 0;

    /* Scan all the residents & copies to identify boundary atoms */ 
    for (i=0; i<n+nbnew; i++) {
      for (kdd=0; kdd<2; kdd++) {
        ku = 2*kd+kdd; /* Neighbor ID */
        /* Add an atom to the boundary-atom list, LSB, for neighbor ku 
           according to bit-condition function, bbd */
        if (bbd(r[i],ku)) lsb[ku][++(lsb[ku][0])] = i;
        abort_if(lsb[ku][0] >= NBMAX-1, "atom_copy: NBMAX exceeded");
      }
    }

    /* Message passing------------------------------------------------*/

    com1=MPI_Wtime(); /* To calculate the communication time */

    /* Loop over the lower & higher directions */
    for (kdd=0; kdd<2; kdd++) {

      inode = nn[ku=2*kd+kdd]; /* Neighbor node ID */

      /* Send & receive the # of boundary atoms-----------------------*/

      nsd = lsb[ku][0]; /* # of atoms to be sent */

      /* Even node: send & recv */
      if (myparity[kd] == 0) {
        MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
                 MPI_COMM_WORLD,&status);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,
                 MPI_COMM_WORLD,&status);
        MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);
      }
      /* Single layer: Exchange information with myself */
      else
        nrc = nsd;
      /* Now nrc is the # of atoms to be received */

      /* Send & receive information on boundary atoms-----------------*/

      abort_if(4*nsd > NDBUF || 4*nrc > NDBUF, "atom_copy: NDBUF exceeded");
      abort_if(n+nbnew+nrc > NEMAX, "atom_copy: NEMAX exceeded");

      /* Message buffering: coordinates & global ID */
      for (i=1; i<=nsd; i++) {
        for (a=0; a<3; a++) /* Shift the coordinate origin */
          dbuf[4*(i-1)+a] = r[lsb[ku][i]][a]-sv[ku][a]; 
        dbuf[4*(i-1)+3] = gid[lsb[ku][i]];
      }

      /* Even node: send & recv */
      if (myparity[kd] == 0) {
        MPI_Send(dbuf,4*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
        MPI_Recv(dbufr,4*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
                 MPI_COMM_WORLD,&status);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
        MPI_Recv(dbufr,4*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,
                 MPI_COMM_WORLD,&status);
        MPI_Send(dbuf,4*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      }
      /* Single layer: Exchange information with myself */
      else
        for (i=0; i<4*nrc; i++) dbufr[i] = dbuf[i];

      /* Message storing */
      for (i=0; i<nrc; i++) {
        for (a=0; a<3; a++) r[n+nbnew+i][a] = dbufr[4*i+a]; 
        gid[n+nbnew+i] = (int)dbufr[4*i+3];
      }

      /* Increment the # of received boundary atoms */
      nbnew = nbnew+nrc;

      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);

    } /* Endfor lower & higher directions, kdd */

    comt += MPI_Wtime()-com1; /* Update communication time, COMT */

  } /* Endfor x, y & z directions, kd */

  /* Main loop over x, y & z directions ends--------------------------*/

  /* Update the # of received boundary atoms */
  nb = nbnew;
}

/*--------------------------------------------------------------------*/
void compute_accel() {
/*----------------------------------------------------------------------
Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e., 
resident & copied) system, computes the acceleration, ra[0:n-1][], and
the per-atom potential energy, pea[0:n-1], for the residents.
----------------------------------------------------------------------*/
  double lpe,t1;

  t1 = MPI_Wtime();
#ifdef USE_CUDA
  lpe = force_cuda(n,nb,&r[0][0],&ra[0][0],pea,lc,rc,RCUT,Uc,Duc);
#else
  lpe = compute_accel_cpu(ra,pea);
#endif
  fct += MPI_Wtime()-t1;

  /* Global potential energy */
  MPI_Allreduce(&lpe,&potEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

/*--------------------------------------------------------------------*/
double compute_accel_cpu(double (*f)[3], double *pe) {
/*----------------------------------------------------------------------
CPU linked-cell force loop.  Returns the local potential energy.
----------------------------------------------------------------------*/
  int i,j,a,lc2[3],lcyz2,lcxyz2,mc[3],c,mc1[3],c1;
  int bintra;
  double dr[3],rr,ri2,ri6,r1,rrCut,fcVal,fa,vVal,lpe;

  /* Reset the potential & forces */
  lpe = 0.0;
  for (i=0; i<n; i++) {
    pe[i] = 0.0;
    for (a=0; a<3; a++) f[i][a] = 0.0;
  }

  /* Make a linked-cell list, lscl------------------------------------*/

  for (a=0; a<3; a++) lc2[a] = lc[a]+2;
  lcyz2 = lc2[1]*lc2[2];
  lcxyz2 = lc2[0]*lcyz2;
  abort_if(lcxyz2 > NCLMAX, "compute_accel: NCLMAX exceeded");

  /* Reset the headers, head */
  for (c=0; c<lcxyz2; c++) head[c] = EMPTY;

  /* Scan atoms to construct headers, head, & linked lists, lscl */

  for (i=0; i<n+nb; i++) {
    for (a=0; a<3; a++) mc[a] = (r[i][a]+rc[a])/rc[a];

    /* Translate the vector cell index, mc, to a scalar cell index */
    c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];

    /* Link to the previous occupant (or EMPTY if you're the 1st) */
    lscl[i] = head[c];

    /* The last one goes to the header */
    head[c] = i;
  } /* Endfor atom i */

  /* Calculate pair interaction---------------------------------------*/

  rrCut = RCUT*RCUT;

  /* Scan inner cells */
  for (mc[0]=1; mc[0]<=lc[0]; (mc[0])++)
  for (mc[1]=1; mc[1]<=lc[1]; (mc[1])++)
  for (mc[2]=1; mc[2]<=lc[2]; (mc[2])++) {

    /* Calculate a scalar cell index */
    c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];
    /* Skip this cell if empty */
    if (head[c] == EMPTY) continue;

    /* Scan the neighbor cells (including itself) of cell c */
    for (mc1[0]=mc[0]-1; mc1[0]<=mc[0]+1; (mc1[0])++)
    for (mc1[1]=mc[1]-1; mc1[1]<=mc[1]+1; (mc1[1])++)
    for (mc1[2]=mc[2]-1; mc1[2]<=mc[2]+1; (mc1[2])++) {

      /* Calculate the scalar cell index of the neighbor cell */
      c1 = mc1[0]*lcyz2+mc1[1]*lc2[2]+mc1[2];
      /* Skip this neighbor cell if empty */
      if (head[c1] == EMPTY) continue;

      /* Scan atom i in cell c */
      i = head[c];
      while (i != EMPTY) {

        /* Scan atom j in cell c1 */
        j = head[c1];
        while (j != EMPTY) {

          /* No calculation with itself */
          if (j != i) {
            /* Logical flag: intra(true)- or inter(false)-pair atom */
            bintra = (j < n);

            /* Pair vector dr = r[i] - r[j] */
            for (rr=0.0, a=0; a<3; a++) {
              dr[a] = r[i][a]-r[j][a];
              rr += dr[a]*dr[a];
            }

            /* Calculate potential & forces for intranode pairs (i < j)
               & all the internode pairs if rij < RCUT; note that for
               any copied atom, i < j */
            if (i<j && rr<rrCut) {
              ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1 = sqrt(rr);
              fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1;
              vVal = 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1-RCUT);
              if (bintra) lpe += vVal; else lpe += 0.5*vVal;
              pe[i] += 0.5*vVal;
              if (bintra) pe[j] += 0.5*vVal;
              for (a=0; a<3; a++) {
                fa = fcVal*dr[a];
                f[i][a] += fa;
                if (bintra) f[j][a] -= fa;
              }
            }
          } /* Endif not self */
          
          j = lscl[j];
        } /* Endwhile j not empty */

        i = lscl[i];
      } /* Endwhile i not empty */

    } /* Endfor neighbor cells, c1 */

  } /* Endfor central cell, c */

  return lpe;
}

/*--------------------------------------------------------------------*/
void eval_props() {
/*----------------------------------------------------------------------
Evaluates physical properties: kinetic, potential & total energies.
----------------------------------------------------------------------*/
  double vv,loc[2],glb[2],pe;
  int i,a;

  /* Total kinetic energy & # of mobile atoms */
  loc[0] = loc[1] = 0.0;
  for (i=0; i<n; i++) {
    for (vv=0.0, a=0; a<3; a++) vv += rv[i][a]*rv[i][a];
    loc[0] += 0.5*vv;
    if (!(seedFrozen && ity[i])) loc[1] += 1.0;
  }
  MPI_Allreduce(loc,glb,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  /* Energy per atom */
  kinEnergy = glb[0]/nglob;
  pe = potEnergy/nglob;
  totEnergy = kinEnergy + pe;
  temperature = 2.0*glb[0]/(3.0*glb[1]);

  /* Print the computed properties */
  if (sid == 0) {
    if (stepCount == 0) printf("#  step      time    Ttarget          T"
                         "             PE             KE             TE\n");
    printf("%7d %9.3f %10.6f %10.6f %14.8f %14.8f %14.8f\n",stepCount,
           stepCount*DeltaT,Thermostat ? target_temp(stepCount) : 0.0,
           temperature,pe,kinEnergy,totEnergy);
  }
}

/*--------------------------------------------------------------------*/
void atom_move() {
/*----------------------------------------------------------------------
Sends moved-out atoms to neighbor nodes and receives moved-in atoms 
from neighbor nodes.  Called with n, r[0:n-1] & rv[0:n-1], atom_move 
returns a new n' together with r[0:n'-1] & rv[0:n'-1].
----------------------------------------------------------------------*/

/* Local variables------------------------------------------------------

mvque[6][NBMAX]: mvque[ku][0] is the # of to-be-moved atoms to neighbor 
  ku; MVQUE[ku][k>0] is the atom ID, used in r, of the k-th atom to be
  moved.
----------------------------------------------------------------------*/
  static int mvque[6][NBMAX];
  int newim = 0; /* # of new immigrants */
  int ku,kd,i,kdd,kul,kuh,inode,ipt,a,nsd,nrc;
  double com1;

  /* Reset the # of to-be-moved atoms, MVQUE[][0] */
  for (ku=0; ku<6; ku++) mvque[ku][0] = 0;

  /* Main loop over x, y & z directions starts------------------------*/

  for (kd=0; kd<3; kd++) {

    /* Make a moved-atom list, mvque----------------------------------*/

    /* Scan all the residents & immigrants to list moved-out atoms */
    for (i=0; i<n+newim; i++) {
      kul = 2*kd  ; /* Neighbor ID */
      kuh = 2*kd+1; 
      /* Register a to-be-copied atom in mvque[kul|kuh][] */      
      if (r[i][0] > MOVED_OUT) { /* Don't scan moved-out atoms */
        /* Move to the lower direction */
        if (bmv(r[i],kul)) mvque[kul][++(mvque[kul][0])] = i;
        /* Move to the higher direction */
        else if (bmv(r[i],kuh)) mvque[kuh][++(mvque[kuh][0])] = i;
      }
      abort_if(mvque[kul][0] >= NBMAX-1 || mvque[kuh][0] >= NBMAX-1,
               "atom_move: NBMAX exceeded");
    }

    /* Message passing with neighbor nodes----------------------------*/

    com1 = MPI_Wtime();

    /* Loop over the lower & higher directions------------------------*/

    for (kdd=0; kdd<2; kdd++) {

      inode = nn[ku=2*kd+kdd]; /* Neighbor node ID */

      /* Send atom-number information---------------------------------*/  

      nsd = mvque[ku][0]; /* # of atoms to-be-sent */

      /* Even node: send & recv */
      if (myparity[kd] == 0) {
        MPI_Send(&nsd,1,MPI_INT,inode,110,MPI_COMM_WORLD);
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,110,
                 MPI_COMM_WORLD,&status);
      }
      /* Odd node: recv & send */
      else if (myparity[kd] == 1) {
        MPI_Recv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,110,
                 MPI_COMM_WORLD,&status);
        MPI_Send(&nsd,1,MPI_INT,inode,110,MPI_COMM_WORLD);
      }
      /* Single layer: Exchange information with myself */
      else
        nrc = nsd;
      /* Now nrc is the # of atoms to be received */

      /* Send & receive information on boundary atoms-----------------*/

      abort_if(8*nsd > NDBUF || 8*nrc > NDBUF, "atom_move: NDBUF exceeded");
      abort_if(n+newim+nrc > NMAX, "atom_move: NMAX exceeded");

      /* Message buffering: coordinates, velocities, global ID & type */
      for (i=1; i<=nsd; i++) {
        for (a=0; a<3; a++) {
          /* Shift the coordinate origin */
          dbuf[8*(i-1)  +a] = r [mvque[ku][i]][a]-sv[ku][a]; 
          dbuf[8*(i-1)+3+a] = rv[mvque[ku][i]][a];
        }
        dbuf[8*(i-1)+6] = gid[mvque[ku][i]];
        dbuf[8*(i-1)+7] = ity[mvque[ku][i]];
        r[mvque[ku][i]][0] = MOVED_OUT; /* Mark the moved-out atom */
      }

      /* Even node: send & recv, if not empty */
      if (myparity[kd] == 0) {
        MPI_Send(dbuf,8*nsd,MPI_DOUBLE,inode,120,MPI_COMM_WORLD);
        MPI_Recv(dbufr,8*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,120,
                 MPI_COMM_WORLD,&status);
      }
      /* Odd node: recv & send, if not empty */
      else if (myparity[kd] == 1) {
        MPI_Recv(dbufr,8*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,120,
                 MPI_COMM_WORLD,&status);
        MPI_Send(dbuf,8*nsd,MPI_DOUBLE,inode,120,MPI_COMM_WORLD);
      }
      /* Single layer: Exchange information with myself */
      else
        for (i=0; i<8*nrc; i++) dbufr[i] = dbuf[i];

      /* Message storing */
      for (i=0; i<nrc; i++) {
        for (a=0; a<3; a++) {
          r [n+newim+i][a] = dbufr[8*i  +a]; 
          rv[n+newim+i][a] = dbufr[8*i+3+a];
        }
        gid[n+newim+i] = (int)dbufr[8*i+6];
        ity[n+newim+i] = (int)dbufr[8*i+7];
      }

      /* Increment the # of new immigrants */
      newim = newim+nrc;

      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);

    } /* Endfor lower & higher directions, kdd */

    comt=comt+MPI_Wtime()-com1;

  } /* Endfor x, y & z directions, kd */
  
  /* Main loop over x, y & z directions ends--------------------------*/

  /* Compress resident arrays including new immigrants */

  ipt = 0;
  for (i=0; i<n+newim; i++) {
    if (r[i][0] > MOVED_OUT) {
      for (a=0; a<3; a++) {
        r [ipt][a] = r [i][a];
        rv[ipt][a] = rv[i][a];
      }
      gid[ipt] = gid[i];
      ity[ipt] = ity[i];
      ++ipt;
    }
  }

  /* Update the compressed # of resident atoms */
  n = ipt;
}

/*----------------------------------------------------------------------
Bit condition functions:

1. bbd(ri,ku) is .true. if coordinate ri[3] is in the boundary to 
     neighbor ku.
2. bmv(ri,ku) is .true. if an atom with coordinate ri[3] has moved out 
     to neighbor ku.
----------------------------------------------------------------------*/
int bbd(double* ri, int ku) {
  int kd,kdd;
  kd = ku/2; /* x(0)|y(1)|z(2) direction */
  kdd = ku%2; /* Lower(0)|higher(1) direction */
  if (kdd == 0)
    return ri[kd] < RCUT;
  else
    return al[kd]-RCUT < ri[kd];
}
int bmv(double* ri, int ku) {
  int kd,kdd;
  kd = ku/2; /* x(0)|y(1)|z(2) direction */
  kdd = ku%2; /* Lower(0)|higher(1) direction */
  if (kdd == 0)
    return ri[kd] < 0.0;
  else
    return al[kd] < ri[kd];
}

/*--------------------------------------------------------------------*/
void write_xyz() {
/*----------------------------------------------------------------------
Gathers all resident atoms to rank 0, which writes an extended XYZ
file <XYZPrefix>_<step>.xyz ordered by global ID.  Columns: ID, type
(0 liquid/normal, 1 seed), x, y, z (global, in [0,L)), per-atom PE.
----------------------------------------------------------------------*/
  static double *loc=NULL,*all=NULL,*rbuf=NULL;
  static int *cnt,*dsp;
  int i,a,k,ntot;
  char fname[512];
  FILE *fp;

  if (loc == NULL) {
    loc = malloc(sizeof(double)*6*NMAX);
    cnt = malloc(sizeof(int)*nproc); dsp = malloc(sizeof(int)*nproc);
    if (sid == 0) {
      all = malloc(sizeof(double)*6*nglob);
      rbuf = malloc(sizeof(double)*6*nglob);
    }
  }
  for (i=0; i<n; i++) {
    loc[6*i] = gid[i]; loc[6*i+1] = ity[i];
    for (a=0; a<3; a++) loc[6*i+2+a] = r[i][a]+vid[a]*al[a];
    loc[6*i+5] = pea[i];
  }
  k = 6*n;
  MPI_Gather(&k,1,MPI_INT,cnt,1,MPI_INT,0,MPI_COMM_WORLD);
  if (sid == 0) for (dsp[0]=0, i=1; i<nproc; i++) dsp[i] = dsp[i-1]+cnt[i-1];
  MPI_Gatherv(loc,k,MPI_DOUBLE,rbuf,cnt,dsp,MPI_DOUBLE,
              0,MPI_COMM_WORLD);
  if (sid != 0) return;

  ntot = (dsp[nproc-1]+cnt[nproc-1])/6;
  abort_if(ntot != nglob, "write_xyz: atom count mismatch");
  for (i=0; i<ntot; i++) {
    k = (int)rbuf[6*i];
    for (a=0; a<6; a++) all[6*k+a] = rbuf[6*i+a];
  }
  sprintf(fname,"%s_%08d.xyz",XYZPrefix,stepCount);
  fp = fopen(fname,"w");
  abort_if(fp == NULL, "write_xyz: cannot open file");
  fprintf(fp,"%d\nLattice=\"%.10f 0 0 0 %.10f 0 0 0 %.10f\" "
          "Properties=id:I:1:type:I:1:pos:R:3:pe:R:1 Step=%d Time=%.4f\n",
          nglob,vproc[0]*al[0],vproc[1]*al[1],vproc[2]*al[2],stepCount,
          stepCount*DeltaT);
  for (i=0; i<nglob; i++)
    fprintf(fp,"%d %d %.8f %.8f %.8f %.8f\n",(int)all[6*i],(int)all[6*i+1],
            all[6*i+2],all[6*i+3],all[6*i+4],all[6*i+5]);
  fclose(fp);
}

#ifdef USE_CUDA
/*--------------------------------------------------------------------*/
void force_check() {
/*----------------------------------------------------------------------
Recomputes the forces of the current configuration on the CPU and
compares them with the GPU result in ra & pea.  Reports the max per-atom
relative force error |dF|/|F| (mobile atoms), max |dF| / max |F|, the
max per-atom relative PE error & the relative error of the total PE.
----------------------------------------------------------------------*/
  static double fc[NMAX][3],pc[NMAX];
  double loc[4],glb[4],sl[2],sg[2],df,fn,d;
  int i,a;

  sl[1] = compute_accel_cpu(fc,pc);
  for (a=0; a<4; a++) loc[a] = 0.0;
  sl[0] = 0.0;
  for (i=0; i<n; i++) {
    for (df=fn=0.0, a=0; a<3; a++) {
      d = ra[i][a]-fc[i][a];
      df += d*d; fn += fc[i][a]*fc[i][a];
    }
    df = sqrt(df); fn = sqrt(fn);
    /* Frozen seed atoms sit at F ~ 0 by symmetry & their F is unused */
    if (!(seedFrozen && ity[i]) && fn > 0.0 && df/fn > loc[0])
      loc[0] = df/fn;
    if (df > loc[1]) loc[1] = df;
    if (fn > loc[2]) loc[2] = fn;
    d = fabs(pea[i]-pc[i])/fabs(pc[i]);
    if (d > loc[3]) loc[3] = d;
    sl[0] += pea[i];
  }
  MPI_Allreduce(loc,glb,4,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(sl,sg,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if (sid == 0)
    printf("FORCECHECK step %d max_rel_f %.3e max_df/max_f %.3e max_f %.3e "
           "max_rel_pe %.3e rel_petot %.3e\n",stepCount,glb[0],
           glb[1]/glb[2],glb[2],glb[3],fabs(sg[0]-sg[1])/fabs(sg[1]));
}
#endif

/*--------------------------------------------------------------------*/
void abort_if(int cond, const char *msg) {
/*--------------------------------------------------------------------*/
  if (cond) {
    fprintf(stderr,"rank %d: %s\n",sid,msg);
    MPI_Abort(MPI_COMM_WORLD,1);
  }
}

#ifdef CHECK_IDS
/*--------------------------------------------------------------------*/
void check_ids() {
/*----------------------------------------------------------------------
Debug check: every global ID is owned by exactly one rank, and every
cached copy sits at a periodic image of the resident with the same ID.
----------------------------------------------------------------------*/
  static double *tab=NULL,*loc,*all;
  static int *cnt,*dsp;
  int i,a,k,bad=0,gbad;
  double d,L;

  if (tab == NULL) {
    tab = malloc(sizeof(double)*3*nglob); all = malloc(sizeof(double)*4*nglob);
    loc = malloc(sizeof(double)*4*NMAX);
    cnt = malloc(sizeof(int)*nproc); dsp = malloc(sizeof(int)*nproc);
  }
  for (i=0; i<n; i++) {
    loc[4*i] = gid[i];
    for (a=0; a<3; a++) loc[4*i+1+a] = r[i][a]+vid[a]*al[a];
  }
  k = 4*n;
  MPI_Allgather(&k,1,MPI_INT,cnt,1,MPI_INT,MPI_COMM_WORLD);
  for (dsp[0]=0, i=1; i<nproc; i++) dsp[i] = dsp[i-1]+cnt[i-1];
  MPI_Allgatherv(loc,k,MPI_DOUBLE,all,cnt,dsp,MPI_DOUBLE,MPI_COMM_WORLD);
  for (i=0; i<3*nglob; i++) tab[i] = MOVED_OUT;
  for (i=0; i<(dsp[nproc-1]+cnt[nproc-1])/4; i++) {
    k = (int)all[4*i];
    if (k < 0 || k >= nglob || tab[3*k] != MOVED_OUT) bad++;
    else for (a=0; a<3; a++) tab[3*k+a] = all[4*i+1+a];
  }
  if ((dsp[nproc-1]+cnt[nproc-1])/4 != nglob) bad++;
  for (i=n; i<n+nb; i++) {
    k = gid[i];
    for (a=0; a<3; a++) {
      L = vproc[a]*al[a];
      d = r[i][a]+vid[a]*al[a]-tab[3*k+a];
      d -= L*floor(d/L+0.5);
      if (fabs(d) > 1e-8) {bad++; break;}
    }
  }
  MPI_Allreduce(&bad,&gbad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (sid == 0) printf("CHECK_IDS step %d errors %d\n",stepCount,gbad);
}
#endif
