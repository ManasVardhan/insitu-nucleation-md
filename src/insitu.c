/*----------------------------------------------------------------------
insitu.c: SIM-side send and ANA-side analysis of the in situ pipeline.
See insitu.h for the protocol and the file formats.
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "insitu.h"
#include "steinhardt.h"
#include "mlp.h"
#include "cluster.h"

#define BIT_MLP 1
#define BIT_THR 2
#define BIT_MAXMLP 4
#define BIT_MAXTHR 8

static void die(const char *msg) {
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD,&r);
  fprintf(stderr,"rank %d: insitu: %s\n",r,msg);
  MPI_Abort(MPI_COMM_WORLD,1);
}

static void *xrealloc(void *p, size_t nbytes) {
  void *q = realloc(p, nbytes > 0 ? nbytes : 1);
  if (q == NULL) die("out of memory");
  return q;
}

int insitu_is_step(const insitu_cfg *c, int step) {
  return c->nana > 0 && step > 0 && step >= c->start && step <= c->last &&
         step%c->interval == 0;
}

int insitu_nsteps(const insitu_cfg *c) {
  int s,k = 0;
  if (c->nana <= 0 || c->interval <= 0) return 0; /* analysis off */
  for (s=c->interval; s<=c->last; s+=c->interval)
    if (insitu_is_step(c,s)) k++;
  return k;
}

/*======================================================================
  SIM side
======================================================================*/
static double *sbuf = NULL;
static size_t scap = 0;
static MPI_Request sreq = MPI_REQUEST_NULL;
static double sim_twait = 0.0, sim_thalo = 0.0, sim_tpack = 0.0;
static double sim_twait_max = 0.0;
static int sim_nsent = 0;

void insitu_sim_send(const insitu_cfg *c, int step, int nres, int ntot,
                     const double *x, const int *g, const double org[3],
                     double t_halo) {
  double t0,tw,tp,*h;
  size_t len;
  int i,a,me,dest;

  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  /* Stall: the previous message must be gone before the buffer is reused */
  t0 = MPI_Wtime();
  MPI_Wait(&sreq,MPI_STATUS_IGNORE);
  tw = MPI_Wtime()-t0;

  t0 = MPI_Wtime();
  len = INSITU_HDR+4*(size_t)ntot;
  if (len > scap) {scap = len+len/4; sbuf = xrealloc(sbuf,scap*sizeof(double));}
  memcpy(sbuf+INSITU_HDR,x,3*(size_t)ntot*sizeof(double));
  for (i=0; i<ntot; i++) sbuf[INSITU_HDR+3*(size_t)ntot+i] = g[i];
  tp = MPI_Wtime()-t0;

  h = sbuf;
  h[0] = step; h[1] = nres; h[2] = ntot;
  for (a=0; a<3; a++) h[3+a] = org[a];
  h[6] = tw; h[7] = t_halo; h[8] = tp; h[9] = me;
  dest = c->nsim+(int)((long)me*c->nana/c->nsim);
  MPI_Isend(sbuf,(int)len,MPI_DOUBLE,dest,INSITU_TAG,MPI_COMM_WORLD,&sreq);

  sim_twait += tw; sim_thalo += t_halo; sim_tpack += tp;
  if (tw > sim_twait_max) sim_twait_max = tw;
  sim_nsent++;
}

void insitu_sim_finish(const insitu_cfg *c, MPI_Comm sim) {
  double t0,tw,loc[5],glb[5];
  int r;

  t0 = MPI_Wtime();
  MPI_Wait(&sreq,MPI_STATUS_IGNORE);
  tw = MPI_Wtime()-t0;
  loc[0] = sim_twait; loc[1] = tw; loc[2] = sim_thalo; loc[3] = sim_tpack;
  loc[4] = sim_twait_max;
  MPI_Reduce(loc,glb,5,MPI_DOUBLE,MPI_MAX,0,sim);
  MPI_Comm_rank(sim,&r);
  if (r == 0)
    printf("INSITU_SIM sends %d stall_total %.4f stall_max %.4f "
           "final_wait %.4f halo_total %.4f pack_total %.4f (max over SIM)\n",
           sim_nsent,glb[0],glb[4],glb[1],glb[2],glb[3]);
  free(sbuf); sbuf = NULL; scap = 0;
}

void insitu_sim_dump(const insitu_cfg *c, MPI_Comm sim, int step, int n,
                     const double *x, const int *g, const double org[3]) {
  static double *loc = NULL,*rbuf = NULL,*all = NULL;
  static int *cnt = NULL,*dsp = NULL,lcap = 0;
  int i,a,k,r,p,ntot,hdr[2];
  char fname[600];
  FILE *fp;

  MPI_Comm_rank(sim,&r); MPI_Comm_size(sim,&p);
  if (n > lcap) {lcap = n+n/4+16; loc = xrealloc(loc,4*sizeof(double)*lcap);}
  if (cnt == NULL) {
    cnt = xrealloc(NULL,sizeof(int)*p); dsp = xrealloc(NULL,sizeof(int)*p);
    if (r == 0) {
      rbuf = xrealloc(NULL,4*sizeof(double)*c->nglob);
      all = xrealloc(NULL,3*sizeof(double)*c->nglob);
    }
  }
  for (i=0; i<n; i++) {
    loc[4*i] = g[i];
    /* Same expression as the ANA side: local + origin */
    for (a=0; a<3; a++) loc[4*i+1+a] = x[3*i+a]+org[a];
  }
  k = 4*n;
  MPI_Gather(&k,1,MPI_INT,cnt,1,MPI_INT,0,sim);
  if (r == 0) for (dsp[0]=0, i=1; i<p; i++) dsp[i] = dsp[i-1]+cnt[i-1];
  MPI_Gatherv(loc,k,MPI_DOUBLE,rbuf,cnt,dsp,MPI_DOUBLE,0,sim);
  if (r != 0) return;

  ntot = (dsp[p-1]+cnt[p-1])/4;
  if (ntot != c->nglob) die("sim_dump: atom count mismatch");
  for (i=0; i<ntot; i++) {
    k = (int)rbuf[4*i];
    for (a=0; a<3; a++) all[3*k+a] = rbuf[4*i+1+a];
  }
  snprintf(fname,sizeof(fname),"%s_full_%08d.bin",c->prefix,step);
  fp = fopen(fname,"wb");
  if (fp == NULL) die("sim_dump: cannot open file");
  hdr[0] = c->nglob; hdr[1] = step;
  fwrite(hdr,sizeof(int),2,fp);
  fwrite(c->box,sizeof(double),3,fp);
  fwrite(all,sizeof(double),3*(size_t)c->nglob,fp);
  fclose(fp);
}

/*======================================================================
  ANA side
======================================================================*/
typedef struct {double gid,x,y,z,bits;} crec; /* gathered crystal record */

static int cmp_crec(const void *a, const void *b) {
  double u = ((const crec*)a)->gid, v = ((const crec*)b)->gid;
  return (u > v)-(u < v);
}

void insitu_ana_run(const insitu_cfg *c, MPI_Comm ana) {
  int me,np,wme,nsrc,src0,s,k,i,j,a,step,nres,ntot,cnt,nst,ist,movie_now;
  int *srcs,*isres = NULL,*coord = NULL,*lab = NULL,*rcnt = NULL,*rdsp = NULL;
  double *rbuf = NULL,*q4 = NULL,*q6 = NULL,*b4 = NULL,*b6 = NULL;
  double *feat = NULL,*pcr = NULL,*mloc = NULL,*mall = NULL;
  double *cpos = NULL,*fd_feat = NULL,*fd_pcr = NULL;
  crec *cloc = NULL,*call = NULL;
  int *fd_id = NULL;
  unsigned char *fd_bits = NULL,*inmax = NULL,*bits_all = NULL;
  float *mpos = NULL;
  size_t rcap = 0,ccap = 0,ncap = 0,fcap = 0,mcap = 0;
  long long bytes = 0,fullbytes = 0;
  double t0,t1,trecv,tfeat,tgat,tclus,tio,tana,tsum = 0.0,tmax = 0.0,tfmax = 0.0;
  double twall0;
  double hl[6],hg[6];
  int ncl,ncr = 0;
  cluster_info im,it;
  char fname[600],row[512];
  FILE *fsum = NULL,*flab = NULL,*fp;
  MPI_Status st;

  MPI_Comm_rank(ana,&me); MPI_Comm_size(ana,&np);
  MPI_Comm_rank(MPI_COMM_WORLD,&wme);
  twall0 = MPI_Wtime();
  if (np != c->nana) die("ANA comm size != InsituRanks");

  /* My SIM sources: s*nana/nsim == me */
  srcs = xrealloc(NULL,sizeof(int)*c->nsim);
  for (nsrc=0, s=0; s<c->nsim; s++)
    if ((int)((long)s*c->nana/c->nsim) == me) srcs[nsrc++] = s;
  src0 = nsrc > 0 ? srcs[0] : -1;
  printf("INSITU_ANA rank %d (world %d) serves %d SIM ranks from %d\n",
         me,wme,nsrc,src0);

  nst = insitu_nsteps(c);
  if (me == 0) {
    rcnt = xrealloc(NULL,sizeof(int)*np); rdsp = xrealloc(NULL,sizeof(int)*np);
    snprintf(fname,sizeof(fname),"%s_summary.dat",c->prefix);
    fsum = fopen(fname,"w");
    snprintf(fname,sizeof(fname),"%s_labels.bin",c->prefix);
    flab = fopen(fname,"wb");
    if (!fsum || !flab) die("cannot open output files");
    k = snprintf(row,sizeof(row),
      "# step time n_crystal_mlp n_crystal_thr nmax_mlp nmax_thr t_analysis "
      "t_stall bytes_insitu | n2_mlp n2_thr nbig_mlp nbig_thr t_feat t_clus_io "
      "t_recvwait t_simhalo t_simpack bytes_fulldump t_wall\n"
      "# nsim %d nana %d K %d nglob %d rq %g thr %g clusmin %d nsteps %d\n",
      c->nsim,c->nana,c->interval,c->nglob,c->rq,c->thr,c->clusmin,nst);
    fputs(row,fsum); bytes += k;
    inmax = xrealloc(NULL,c->nglob);
    if (c->movie > 0) {
      mpos = xrealloc(NULL,3*sizeof(float)*c->nglob);
      bits_all = xrealloc(NULL,c->nglob);
    }
  }

  for (ist=0, step=c->interval; step<=c->last; step+=c->interval) {
    if (!insitu_is_step(c,step)) continue;
    ist++;
    movie_now = (c->movie > 0 && ist%c->movie == 0);
    trecv = tfeat = 0.0;
    hl[0] = hl[1] = hl[2] = 0.0; /* max stall, halo, pack over my SIMs */
    ncl = 0;
    int nmloc = 0, nfd = 0;

    for (k=0; k<nsrc; k++) {
      /* Receive the block of SIM rank srcs[k] */
      t0 = MPI_Wtime();
      MPI_Probe(srcs[k],INSITU_TAG,MPI_COMM_WORLD,&st);
      MPI_Get_count(&st,MPI_DOUBLE,&cnt);
      if ((size_t)cnt > rcap) {rcap = cnt+cnt/4; rbuf = xrealloc(rbuf,rcap*sizeof(double));}
      MPI_Recv(rbuf,cnt,MPI_DOUBLE,srcs[k],INSITU_TAG,MPI_COMM_WORLD,&st);
      trecv += MPI_Wtime()-t0;

      t0 = MPI_Wtime();
      if ((int)rbuf[0] != step) die("step mismatch in SIM message");
      nres = (int)rbuf[1]; ntot = (int)rbuf[2];
      if (cnt != INSITU_HDR+4*ntot) die("bad SIM message size");
      for (a=0; a<3; a++) if (rbuf[6+a] > hl[a]) hl[a] = rbuf[6+a];
      const double *pos = rbuf+INSITU_HDR, *gd = rbuf+INSITU_HDR+3*(size_t)ntot;
      const double *org = rbuf+3;

      if ((size_t)ntot > ncap) {
        ncap = ntot+ntot/4;
        isres = xrealloc(isres,sizeof(int)*ncap); coord = xrealloc(coord,sizeof(int)*ncap);
        lab = xrealloc(lab,sizeof(int)*ncap);
        q4 = xrealloc(q4,sizeof(double)*ncap); q6 = xrealloc(q6,sizeof(double)*ncap);
        b4 = xrealloc(b4,sizeof(double)*ncap); b6 = xrealloc(b6,sizeof(double)*ncap);
        feat = xrealloc(feat,STEINHARDT_NFEAT*sizeof(double)*ncap);
        pcr = xrealloc(pcr,sizeof(double)*ncap);
      }
      for (i=0; i<ntot; i++) isres[i] = (i < nres);
      if (steinhardt_compute(ntot,pos,isres,c->rq,q4,q6,b4,b6,coord) != 0)
        die("steinhardt_compute failed");
      for (i=0; i<nres; i++)
        steinhardt_features(q4[i],q6[i],b4[i],b6[i],coord[i],feat+5*(size_t)i);
      mlp_classify_batch(nres,feat,lab,pcr);

      /* Crystalline residents -> gather list (global coordinates) */
      if ((size_t)(ncl+nres) > ccap) {
        ccap = ncl+nres+(ncl+nres)/4;
        cloc = xrealloc(cloc,sizeof(crec)*ccap);
      }
      if (c->check && (size_t)(nfd+nres) > fcap) {
        fcap = nfd+nres+(nfd+nres)/4;
        fd_id = xrealloc(fd_id,sizeof(int)*fcap);
        fd_feat = xrealloc(fd_feat,5*sizeof(double)*fcap);
        fd_pcr = xrealloc(fd_pcr,sizeof(double)*fcap);
        fd_bits = xrealloc(fd_bits,fcap);
      }
      if (movie_now && (size_t)(nmloc+nres) > mcap) {
        mcap = nmloc+nres+(nmloc+nres)/4;
        mloc = xrealloc(mloc,4*sizeof(double)*mcap);
      }
      for (i=0; i<nres; i++) {
        int b = (lab[i] == MLP_LABEL_CRYSTAL ? BIT_MLP : 0) |
                (b6[i] > c->thr ? BIT_THR : 0);
        if (b) {
          cloc[ncl].gid = gd[i]; cloc[ncl].bits = b;
          cloc[ncl].x = pos[3*i]+org[0];
          cloc[ncl].y = pos[3*i+1]+org[1];
          cloc[ncl].z = pos[3*i+2]+org[2];
          ncl++;
        }
        if (c->check) {
          fd_id[nfd] = (int)gd[i];
          memcpy(fd_feat+5*(size_t)nfd,feat+5*(size_t)i,5*sizeof(double));
          fd_pcr[nfd] = pcr[i]; fd_bits[nfd] = b; nfd++;
        }
        if (movie_now) {
          mloc[4*nmloc] = gd[i];
          for (a=0; a<3; a++) mloc[4*nmloc+1+a] = pos[3*i+a]+org[a];
          nmloc++;
        }
      }
      tfeat += MPI_Wtime()-t0;
    }

    /* Equivalence-check feature dump, one file per ANA rank */
    if (c->check) {
      snprintf(fname,sizeof(fname),"%s_feat_%08d_a%d.bin",c->prefix,step,me);
      fp = fopen(fname,"wb");
      if (fp == NULL) die("cannot open feature dump");
      fwrite(&nfd,sizeof(int),1,fp);
      fwrite(fd_id,sizeof(int),nfd,fp);
      fwrite(fd_feat,sizeof(double),5*(size_t)nfd,fp);
      fwrite(fd_pcr,sizeof(double),nfd,fp);
      fwrite(fd_bits,1,nfd,fp);
      fclose(fp);
    }

    /* Gather crystalline atoms to ANA root */
    t0 = MPI_Wtime();
    k = 5*ncl;
    MPI_Gather(&k,1,MPI_INT,rcnt,1,MPI_INT,0,ana);
    if (me == 0) {
      for (rdsp[0]=0, i=1; i<np; i++) rdsp[i] = rdsp[i-1]+rcnt[i-1];
      ncr = (rdsp[np-1]+rcnt[np-1])/5;
      call = xrealloc(call,sizeof(crec)*(ncr > 0 ? ncr : 1));
    }
    MPI_Gatherv(cloc,k,MPI_DOUBLE,call,rcnt,rdsp,MPI_DOUBLE,0,ana);
    tgat = MPI_Wtime()-t0;

    /* Max over ANA ranks: SIM stall/halo/pack, feature time, busy time */
    hl[3] = tfeat; hl[4] = tfeat+tgat; hl[5] = trecv;
    MPI_Reduce(hl,hg,6,MPI_DOUBLE,MPI_MAX,0,ana);

    if (movie_now) {
      k = 4*nmloc;
      MPI_Gather(&k,1,MPI_INT,rcnt,1,MPI_INT,0,ana);
      if (me == 0) {
        for (rdsp[0]=0, i=1; i<np; i++) rdsp[i] = rdsp[i-1]+rcnt[i-1];
        if (rdsp[np-1]+rcnt[np-1] != 4*c->nglob) die("movie: atom count mismatch");
        if (mall == NULL) mall = xrealloc(NULL,4*sizeof(double)*c->nglob);
      }
      MPI_Gatherv(mloc,k,MPI_DOUBLE,mall,rcnt,rdsp,MPI_DOUBLE,0,ana);
    }

    if (me == 0) {
      /* Largest clusters for both label sets, atoms in ID order */
      t1 = MPI_Wtime();
      qsort(call,ncr,sizeof(crec),cmp_crec);
      cpos = xrealloc(cpos,3*sizeof(double)*(ncr > 0 ? ncr : 1));
      for (a=0; a<2; a++) {
        int bit = a == 0 ? BIT_MLP : BIT_THR, m = 0;
        for (i=0; i<ncr; i++)
          if ((int)call[i].bits & bit) {
            cpos[3*m] = call[i].x; cpos[3*m+1] = call[i].y; cpos[3*m+2] = call[i].z;
            m++;
          }
        if (cluster_find(m,cpos,c->box,c->rq,c->clusmin,inmax,a == 0 ? &im : &it) != 0)
          die("cluster_find failed");
        for (m=0, i=0; i<ncr; i++)
          if ((int)call[i].bits & bit) {
            if (inmax[m]) call[i].bits = (int)call[i].bits | (a == 0 ? BIT_MAXMLP : BIT_MAXTHR);
            m++;
          }
      }
      tclus = MPI_Wtime()-t1;

      /* Label file */
      t1 = MPI_Wtime();
      {
        int hdr[2] = {step,ncr};
        int *ids = xrealloc(NULL,sizeof(int)*(ncr > 0 ? ncr : 1));
        unsigned char *bb = xrealloc(NULL,ncr > 0 ? ncr : 1);
        for (i=0; i<ncr; i++) {ids[i] = (int)call[i].gid; bb[i] = (int)call[i].bits;}
        fwrite(hdr,sizeof(int),2,flab);
        fwrite(ids,sizeof(int),ncr,flab);
        fwrite(bb,1,ncr,flab);
        fflush(flab);
        bytes += 8+5*(long long)ncr;
        free(ids); free(bb);
      }
      /* Movie snapshot */
      if (movie_now) {
        int hdr[2] = {c->nglob,step};
        double tm = step*c->dt;
        memset(bits_all,0,c->nglob);
        for (i=0; i<ncr; i++) bits_all[(int)call[i].gid] = (int)call[i].bits;
        for (i=0; i<c->nglob; i++) {
          j = (int)mall[4*i];
          for (a=0; a<3; a++) {
            double L = c->box[a], v = mall[4*i+1+a];
            v -= L*floor(v/L);
            if (v >= L) v -= L;
            mpos[3*j+a] = (float)v;
          }
        }
        snprintf(fname,sizeof(fname),"%s_movie_%08d.bin",c->prefix,step);
        fp = fopen(fname,"wb");
        if (fp == NULL) die("cannot open movie file");
        fwrite(hdr,sizeof(int),2,fp);
        fwrite(&tm,sizeof(double),1,fp);
        fwrite(c->box,sizeof(double),3,fp);
        fwrite(mpos,sizeof(float),3*(size_t)c->nglob,fp);
        fwrite(bits_all,1,c->nglob,fp);
        fclose(fp);
        bytes += 8+32+13*(long long)c->nglob;
      }
      tio = MPI_Wtime()-t1;

      tana = hg[4]+tclus+tio;
      tsum += tana; if (tana > tmax) tmax = tana;
      if (hg[3] > tfmax) tfmax = hg[3];
      fullbytes += INSITU_FULL_HDR+24*(long long)c->nglob;
      /* Fill the crystal counts and the byte count including this row */
      {
        int nm = 0,nt = 0;
        for (i=0; i<ncr; i++) {
          if ((int)call[i].bits & BIT_MLP) nm++;
          if ((int)call[i].bits & BIT_THR) nt++;
        }
        long long b2 = bytes;
        for (;;) { /* row length depends on the digits of b2 */
          k = snprintf(row,sizeof(row),
            "%d %.4f %d %d %d %d %.6f %.6f %lld | %d %d %d %d %.6f %.6f %.6f "
            "%.6f %.6f %lld %.3f\n",step,step*c->dt,nm,nt,im.nmax,it.nmax,tana,hg[0],
            b2,im.n2,it.n2,im.nbig,it.nbig,hg[3],tclus+tio,hg[5],hg[1],hg[2],
            fullbytes,MPI_Wtime()-twall0);
          if (bytes+k == b2) break;
          b2 = bytes+k;
        }
      }
      fputs(row,fsum); fflush(fsum);
      bytes += k;
    }
  }

  if (me == 0) {
    fclose(fsum); fclose(flab);
    printf("INSITU_ANA steps %d t_analysis_total %.3f mean %.4f max %.4f "
           "t_feat_max %.4f bytes_insitu %lld bytes_fulldump %lld ratio %.1f\n",
           nst,tsum,nst > 0 ? tsum/nst : 0.0,tmax,tfmax,bytes,fullbytes,
           bytes > 0 ? (double)fullbytes/bytes : 0.0);
  }
  free(srcs); free(isres); free(coord); free(lab); free(rcnt); free(rdsp);
  free(rbuf); free(q4); free(q6); free(b4); free(b6); free(feat); free(pcr);
  free(mloc); free(mall); free(cpos); free(fd_feat); free(fd_pcr); free(cloc);
  free(call); free(fd_id); free(fd_bits); free(inmax); free(bits_all); free(mpos);
}
