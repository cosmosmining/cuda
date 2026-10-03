# 18-647 HW2 -- build with any C11 compiler that supports OpenMP (gcc >= 7, clang + libomp).
#   make            build ./reorder
#   make test       build and run the correctness suite
#   make bench      full measurement (see run_all.sh)
ifeq ($(origin CC),default)
CC      = gcc
endif
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -Wno-unknown-pragmas
OMPFLAG ?= -fopenmp

reorder: main.c reorder.c reorder.h
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ main.c reorder.c

test: reorder
	./reorder test

bench: reorder
	./run_all.sh

clean:
	rm -f reorder

.PHONY: test bench clean
