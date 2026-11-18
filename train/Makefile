# Training-data generator (serial LJ MD + src/steinhardt.c).
CC     ?= cc
CFLAGS ?= -std=c99 -O3 -Wall -Wextra
SRC    := ../src

gen_data: gen_data.c $(SRC)/steinhardt.c $(SRC)/steinhardt.h
	$(CC) $(CFLAGS) -I$(SRC) -o $@ gen_data.c $(SRC)/steinhardt.c -lm

.PHONY: clean
clean:
	rm -f gen_data
