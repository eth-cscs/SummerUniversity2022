CC=gcc
#CFLAGS=-g -fsanitize=address
CFLAGS=-O3 -fno-inline
LIKWID_FLAGS=$(LIKWID_INC) $(LIKWID_LIB)
LLIKWID=-llikwid

all: pagerank_csr_likwid

pagerank_csr: mmio.c pagerank_csr.c
	$(CC) $(CFLAGS) $(LIKWID_FLAGS) -I. $^ -o $@ -fopenmp -lm $(LLIKWID)

clean:
	rm -f pagerank_csr

