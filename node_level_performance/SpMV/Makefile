COMPILER ?= gcc

ifeq ($(COMPILER),gcc)
  CXX       = g++

  # OPT_LEVEL = -O0 -g -ggdb3
  # OPT_LEVEL = -Ofast -g -ggdb3
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp $(OPT_ARCH)

  ifeq ($(ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
  endif

  ifeq ($(UBSAN),1)
    CXXFLAGS += -fsanitize=undefined -g
  endif

else
ifeq ($(COMPILER),clang)
  CXX       = clang++

  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -fopenmp $(OPT_ARCH)

  ifeq ($(ASAN),1)
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
  endif

  ifeq ($(UBSAN),1)
    CXXFLAGS += -fsanitize=undefined,unsigned-integer-overflow,local-bounds,implicit-conversion -g
  endif

else
ifeq ($(COMPILER),intel)
  CXX       = icpc

  OPT_LEVEL = -Ofast
  OPT_ARCH  = -xHost

  CXXFLAGS += $(OPT_LEVEL) -std=c++14 -Wall -qopenmp $(OPT_ARCH)
  # CXXFLAGS += -Wcheck -Wformat -Wformat-security -Wuninitialized -Wpointer-arith -Wshadow -Woverflow

else
# TODO: implement intel-ng
  $(error Unsupported COMPILER specified.)
endif
endif
endif

ifeq ($(LISTINGS),1)
  CXXFLAGS_LISTINGS += -Wa,-alnds=$@.lst -g
  ifeq ($(COMPILER),clang)
    CXXFLAGS_LISTINGS += -no-integrated-as
  else
  ifeq ($(COMPILER),intel-ng)
    CXXFLAGS_LISTINGS += -no-integrated-as
  endif
  endif
endif


NVCC       = nvcc
NVCCFLAGS += -O3 -arch=sm_60
NVCCFLAGS += --use_fast_math
NVCCFLAGS += --fassociative-math --extra-device-vectorization
NVCCFLAGS += -std=c++14

# for profiling/debugging
# NVCCFLAGS += -lineinfo -g

# Host compiler GCC/clang:
# NVCCFLAGS += -Xcompiler -Ofast,-march=native
# Host compiler Intel
# NVCCFLAGS += -Xcompiler -Ofast,-xHost

NVCCLIBS   =

ifeq ($(USE_LIKWID),1)
  CXXFLAGS  += -DUSE_LIKWID $(LIKWID_INC) $(LIKWID_LIB)
  NVCCFLAGS += -DUSE_LIKWID $(LIKWID_INC) $(LIKWID_LIB)
  LIBS      += -llikwid
endif

# Also rebuild when following files change.
REBUILD_DEPS = $(MAKEFILE_LIST) spmv.h vectors.h mtx-reader.h

.PHONY: all
all: gpu omp

.PHONY: omp gpu
omp: spmv-omp
gpu: spmv-gpu


spmv-gpu: main-gpu.o spmv-gpu.o spmv-gpu-adv.o $(REBUILD_DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

main-gpu.o: main.cpp test.cpp $(REBUILD_DEPS)
	$(NVCC) $(NVCCFLAGS) -x cu -o $@ -c $<

spmv-gpu.o: spmv-gpu.cu $(REBUILD_DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

spmv-gpu-adv.o: spmv-gpu-adv.cu $(REBUILD_DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<



spmv-omp: main-omp.o spmv-omp.o spmv-omp-adv.o $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter-out $(REBUILD_DEPS),$^) $(LIBS)

main-omp.o: main.cpp test.cpp $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

spmv-omp.o: spmv-omp.cpp $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CXXFLAGS_LISTINGS)

spmv-omp-adv.o: spmv-omp-adv.cpp $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(CXXFLAGS_LISTINGS)


.PHONY: clean
clean:
	-rm *.o
	-rm spmv-gpu spmv-omp


