#include <iostream>

#include <cuda.h>

#include "util.hpp"

// Downsampling kernel using Shared Memory
template <class T, int BSIZE>
__global__
void downsample_shared(const T* in, T* out, size_t n, int DECIMATE)
{
    // Allocate shared memory statically
    __shared__ T buffer[BSIZE];
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Coalesced reads - no bank conflicts
    if (idx * DECIMATE < n)
    {
        buffer[threadIdx.x] = in[idx * DECIMATE];
    }
    __syncthreads();
    
    // Coalesced writes
    if (idx < n/DECIMATE)
    {
        out[idx] = buffer[threadIdx.x];
    }

}

// Downsample kernel direct
template <class T>
__global__
void downsample(const T* in, T* out, size_t n, int DECIMATE)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx * DECIMATE < n)
    {
        out[idx] = in[idx * DECIMATE];
    }
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    int DECIMATE = 2;
    auto size_in_bytes = n * sizeof(float);
    auto size_in_bytes_out = n/DECIMATE * sizeof(float);

    cuInit(0);

    std::cout << "memcopy test of size " << n << "\n";

    float* in_device = malloc_device<float>(n);
    float* out_device = malloc_device<float>(n/DECIMATE);
    float* out_device_sm = malloc_device<float>(n/DECIMATE);

    float* in_host = malloc_host<float>(n, 1.5);
    float* out_host = malloc_host<float>(n/DECIMATE, 0.0);
    float* out_host_sm = malloc_host<float>(n/DECIMATE, 0.0);

    // copy to device
    auto start = get_time();
    copy_to_device<float>(in_host, in_device, n);
    auto time_H2D = get_time() - start;

    // calculate grid dimensions 
    // NOTE to update threads_per_block in the shared memory kernel call
    int threads_per_block = 32;
    int num_blocks = (n + threads_per_block - 1)/threads_per_block;

    // synchronize the host and device so that the timings are accurate
    cudaDeviceSynchronize();

    // Without Shared Memory
    start = get_time();
    // TODO launch kernel
    downsample<<<num_blocks, threads_per_block>>>(in_device, out_device, n, DECIMATE);

    cudaDeviceSynchronize();
    auto time_downsample = get_time() - start;

    // check for error in last kernel call
    cuda_check_last_kernel("downsample kernel");

    // copy result back to host
    start = get_time();
    copy_to_host<float>(out_device, out_host, n/DECIMATE);
    auto time_D2H = get_time() - start;

    // WITH Shared Memory
    start = get_time();
    // TODO launch kernel
    downsample_shared<float, 32><<<num_blocks, threads_per_block>>>(in_device, out_device_sm, n, DECIMATE);

    cudaDeviceSynchronize();
    auto time_downsample_sm = get_time() - start;

    // check for error in last kernel call
    cuda_check_last_kernel("downsample_shared kernel");

    // copy result back to host
    copy_to_host<float>(out_device_sm, out_host_sm, n/DECIMATE);

    std::cout << "-------\ntimings\n-------\n";
    std::cout << "H2D:  " << time_H2D << " s\n";
    std::cout << "D2H:  " << time_D2H << " s\n";
    std::cout << std::endl;
    std::cout << "downsample: " << time_downsample << " s\n";
    std::cout << "downsample_shared: " << time_downsample_sm << " s\n";
    
    std::cout << std::endl;

    std::cout << "-------\nbandwidth\n-------\n";
    auto H2D_BW = size_in_bytes/1e6*2 / time_H2D;
    auto D2H_BW = size_in_bytes_out/1e6   / time_D2H;
    std::cout << "H2D BW:  " << H2D_BW << " MB/s\n";
    std::cout << "D2H BW:  " << D2H_BW << " MB/s\n";

    // check for errors
    auto errors = 0;
    for(auto i=0; i<n/DECIMATE; ++i) {
        if(std::fabs(1.5-out_host[i])>1e-15) {
            ++errors;
        }
    }

    std::cout << (errors>0 ? "failed" : "passed") << " with " << errors << " errors\n";

    // check for errors
    errors = 0;
    for(auto i=0; i<n/DECIMATE; ++i) {
        if(std::fabs(1.5-out_host_sm[i])>1e-15) {
            ++errors;
        }
    }

    std::cout << (errors>0 ? "failed" : "passed") << " with " << errors << " errors\n";

    cudaFree(in_device);
    cudaFree(out_device);
    cudaFree(out_device_sm);

    free(in_host);
    free(out_host);
    free(out_host_sm);

    return 0;
}

