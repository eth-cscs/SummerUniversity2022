## CUDA Lesson Plan

### Day 1

#### Introduction

Understand motivations behind using GPUs for HPC. Key architectural distinctions between CPU & GPU.

`01_introduction.pdf`

#### CUDA API

Learn the programming model, common GPU libraries, understand GPU memory management with practical exercises. 

`02_porting.pdf`

`03_runtime_api.pdf`

Exercises under `practicals/api` folder.

#### CUDA Kernels

Getting started with writing custom GPU kernels. 

`04_kernels.pdf`

### Day 2

#### Kernels & Threads

Writing custom GPU kernels, understanding concepts of CUDA threads, blocks and grids with practical exercises  

`04_kernels.pdf`

Exercises under `practicals/axpy` folder.

#### Shared Memory and Block Syncronization

Learn using cooperating thread blocks for more advanced kernels. Understand concepts such as race conditions, thread synchronization, atomics with practical exercises. 

`05_shared.pdf`

Exercises under `practicals/shared` folder.

#### CUDA 2D

Learn to use the CUDA api for data in 2D arrays. Useful for many common scientific applications.

`06_cuda2d.pdf`

Exercises under `practicals/diffusion` folder.

### Day 3

#### 2D Diffusion Miniapp

Understand implementing a real-world numerical simulation using a toy mini-app. Leverage previous concepts to implement working GPU code, and compare with a CPU version. The same example would be extended for future lessons on OpenACC as well.

`07_miniapp_intro.pdf`

`08_miniapp.pdf`

Coding exercises in the top level `/miniapp/cuda` folder. Contains a working `OpenMP` implementation as well in the `miniapp/openmp` folder.

#### Bonus Content: Advanced GPU Concepts

Asynchronous operations for concurrency, and using GPUs in distributed computing. Will not be covered in lectures, but extra content for motivated learners with practical examples. 

`async.pdf`

`cuda_mpi.pdf`

Exercises under `practicals/async` folder.

#### NOTE: Solutions would be uploaded in the end of the day in the same repo in the `solutions/` folder.