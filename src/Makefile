# Build the parallel MD code.
#   make pmd            CPU force path (mpicc only)
#   make pmd_cuda       CUDA force path (-DUSE_CUDA, nvcc for force_cuda.cu)
#   make pmd_insitu     CUDA force path + in situ analysis ranks (-DINSITU)
#   make pmd_insitu_cpu CPU force path + in situ analysis (local testing)
#   make posthoc        post hoc equivalence check (plain C, no MPI)
# ANA ranks use OpenMP threads in steinhardt.c; on a compiler without
# OpenMP (Apple clang) build with OMPFLAGS= .
# On CARC Discovery:  module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3

MPICC  ?= mpicc
NVCC   ?= nvcc
CFLAGS ?= -O3 -Wall -Wno-unused-result
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which $(NVCC)))
# P100, V100, A100, A40, L40S
GENCODE ?= -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89

OMPFLAGS ?= -fopenmp
CC ?= cc
ANA_SRC = insitu.c steinhardt.c mlp.c cluster.c
ANA_HDR = insitu.h steinhardt.h mlp.h mlp_weights.h cluster.h

all: pmd pmd_cuda pmd_insitu posthoc

pmd: pmd.c pmd.h
	$(MPICC) $(CFLAGS) -o $@ pmd.c -lm

pmd_cuda: pmd_cuda.o force_cuda.o
	$(MPICC) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -lpthread -lrt -lm

pmd_cuda.o: pmd.c pmd.h force_cuda.h
	$(MPICC) $(CFLAGS) -DUSE_CUDA -c -o $@ pmd.c

force_cuda.o: force_cuda.cu force_cuda.h
	$(NVCC) -O3 $(GENCODE) -Wno-deprecated-gpu-targets -c -o $@ force_cuda.cu

pmd_insitu: pmd_insitu.o force_cuda.o $(ANA_SRC) $(ANA_HDR)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ pmd_insitu.o force_cuda.o $(ANA_SRC) \
	  -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -lpthread -lrt -lm

pmd_insitu.o: pmd.c pmd.h force_cuda.h insitu.h
	$(MPICC) $(CFLAGS) -DUSE_CUDA -DINSITU -c -o $@ pmd.c

pmd_insitu_cpu: pmd.c pmd.h $(ANA_SRC) $(ANA_HDR)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -DINSITU -o $@ pmd.c $(ANA_SRC) -lm

posthoc: posthoc.c steinhardt.c mlp.c cluster.c steinhardt.h mlp.h mlp_weights.h cluster.h
	$(CC) -O3 -std=c99 -Wall $(OMPFLAGS) -o $@ posthoc.c steinhardt.c mlp.c cluster.c -lm

clean:
	rm -f pmd pmd_cuda pmd_insitu pmd_insitu_cpu posthoc *.o

.PHONY: all clean
