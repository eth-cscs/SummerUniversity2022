# Kokkos (Friday 15.07.2022)

This README contains pointers to the materials used in the summer school.

We will be using material prepared by the Kokkos team. We will be using commit
[`f7193c8c5be143f81e933ec70d1784620aa869cd` of the kokkos-tutorials
repository](https://github.com/kokkos/kokkos-tutorials/tree/f7193c8c5be143f81e933ec70d1784620aa869cd).

## Slides

We will primarily be using the ["Short" tutorial
material](https://github.com/kokkos/kokkos-tutorials/blob/f7193c8c5be143f81e933ec70d1784620aa869cd/Intro-Short/KokkosTutorial_Short.pdf),
with selected material from the ["Medium"
tutorial](https://github.com/kokkos/kokkos-tutorials/blob/f7193c8c5be143f81e933ec70d1784620aa869cd/Intro-Medium/KokkosTutorial_Medium.pdf)
subject to time.

## Exercises

The slides have pointers to appropriate exercises. All exercises can be found in
the [Exercises
subdirectory](https://github.com/kokkos/kokkos-tutorials/tree/f7193c8c5be143f81e933ec70d1784620aa869cd/Exercises).

Kokkos can be built and used with CMake or make. For regular use we recommend
using CMake and installing Kokkos with spack. You're free to use whichever setup
you find easiest. However, for the summer school we recommend using the simple
Makefiles-based setup as it gets you started quickly on Piz Daint. The following
instructions are for use on Piz Daint.

Create a directory for Kokkos and the tutorial material:

``` sh
mkdir -p ~/Kokkos
```

Clone the Kokkos and tutorials repositories:

``` sh
git clone https://github.com/kokkos/kokkos.git ~/Kokkos/kokkos
git clone https://github.com/kokkos/kokkos-tutorials.git ~/Kokkos/kokkos-tutorials
```

Load the correct modules (we will be using GCC 9.3 with CUDA 11.2 since the
exercises assume the use of `g++` and `nvcc`; however, other combinations are
also possible):

``` sh
module load daint-gpu
module load cudatoolkit
module switch PrgEnv-cray/6.0.10 PrgEnv-gnu
module switch gcc/11.2.0 gcc/9.3.0
```

All exercises have a `Begin` and a `Solution` directory. The `Begin` directory
contains an exercise which requires some modification. The parts which need to
be changed or added in are marked with `EXERCISE` to make finding them easier.
The `Solution` directory contains a working solution.

All exercises come with a Makefile which is set up to use and build Kokkos
automatically from the cloned repository. To build e.g. the first exercise,
first go to the directory and then build it, specifying the architecture that we
want to build for (Haswell and P100 on Piz Daint):

``` sh
cd ~/Kokkos/kokkos-tutorials/Exercises/01/Begin
make KOKKOS_ARCH=HSW,Pascal60 -j
```

The exercises have appropriate defaults for which backends ("devices") to use.
To change the default, you can specify the `KOKKOS_DEVICES` option. To
explicitly build with the OpenMP and CUDA backends, set it like this:

``` sh
make KOKKOS_ARCH=HSW,Pascal60 KOKKOS_DEVICES=OpenMP,Cuda -j
```

Finally, get an allocation on the GPU partition and run the exercise or
solution. Binaries compiled for the host only have a `.host` extension, while
binaries compiled for CUDA have a `.cuda` extension.

``` sh
salloc -A class06 -C gpu
srun ./01_Exercise.host
```
