CC=gcc
#CFLAGS=-g -fsanitize=address
CFLAGS=-O3 -fno-inline -pg

all: pagerank_csc pagerank_csr

pagerank_csc: mmio.c pagerank_csc.c
	$(CC) $(CFLAGS) -I. $^ -o $@ -fopenmp -lm

pagerank_csr: mmio.c pagerank_csr.c
	$(CC) $(CFLAGS) -I. $^ -o $@ -fopenmp -lm	

clean:
	rm -f pagerank_csc  pagerank_csr

