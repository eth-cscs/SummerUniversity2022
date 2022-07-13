FLAGS=-arch=sm_80 -std=c++11 -O2 -lineinfo -I../include
#sm_60 for P100
LINK=-lcublas -lcuda

BENCHMARKS=diffusion2d

all : $(BENCHMARKS)

# diffusion2D benchmark
diffusion2d : diffusion2d.cu
	nvcc $(FLAGS) diffusion2d.cu $(LINK) -o diffusion2d

clean :
	rm -f ${BENCHMARKS}
	rm -f *.o
	rm -f output.*
