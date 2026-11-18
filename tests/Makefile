# Unit tests for the analysis kernels in src/.
#   make test          build and run all tests
#   make CC=mpicc test also works (no MPI is used)
CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pedantic
SRC     := ../src
LDLIBS  := -lm

TESTS := test_steinhardt test_mlp test_cluster

.PHONY: all test clean
all: $(TESTS)

test_steinhardt: test_steinhardt.c $(SRC)/steinhardt.c $(SRC)/steinhardt.h
	$(CC) $(CFLAGS) -I$(SRC) -o $@ test_steinhardt.c $(SRC)/steinhardt.c $(LDLIBS)

test_mlp: test_mlp.c $(SRC)/mlp.c $(SRC)/mlp.h $(SRC)/mlp_weights.h
	$(CC) $(CFLAGS) -I$(SRC) -o $@ test_mlp.c $(SRC)/mlp.c $(LDLIBS)

test_cluster: test_cluster.c $(SRC)/cluster.c $(SRC)/cluster.h
	$(CC) $(CFLAGS) -I$(SRC) -o $@ test_cluster.c $(SRC)/cluster.c $(LDLIBS)

test: $(TESTS)
	./test_steinhardt
	./test_mlp data/mlp_ref.csv
	./test_cluster

clean:
	rm -f $(TESTS)
