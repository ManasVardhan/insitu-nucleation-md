# Build the parallel MD code.
#   make pmd        CPU force path (mpicc only)
#   make pmd_cuda   CUDA force path (-DUSE_CUDA, nvcc for force_cuda.cu)
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

all: pmd pmd_cuda

pmd: pmd.c pmd.h
	$(MPICC) $(CFLAGS) -o $@ pmd.c -lm

pmd_cuda: pmd_cuda.o force_cuda.o
	$(MPICC) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -lpthread -lrt -lm

pmd_cuda.o: pmd.c pmd.h force_cuda.h
	$(MPICC) $(CFLAGS) -DUSE_CUDA -c -o $@ pmd.c

force_cuda.o: force_cuda.cu force_cuda.h
	$(NVCC) -O3 $(GENCODE) -Wno-deprecated-gpu-targets -c -o $@ force_cuda.cu

clean:
	rm -f pmd pmd_cuda *.o

.PHONY: all clean
