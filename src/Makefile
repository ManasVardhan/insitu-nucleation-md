# Build the parallel MD code.
#   make pmd        CPU force path (mpicc only)
#   make pmd_cuda   CUDA force path (-DUSE_CUDA, nvcc for force_cuda.cu)
# On CARC Discovery:  module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3

MPICC  ?= mpicc
CFLAGS ?= -O3 -Wall -Wno-unused-result

all: pmd

pmd: pmd.c pmd.h
	$(MPICC) $(CFLAGS) -o $@ pmd.c -lm

clean:
	rm -f pmd pmd_cuda *.o

.PHONY: all clean
