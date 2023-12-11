#include <math.h>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;


void getDeviceProperty() {

    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    printf("The number of devices available is %i GPUs \n", count);
    cudaGetDeviceProperties(&prop, count - 1);
    printf("Name: %s\n", prop.name);
    printf("Global memory size in bytes: %ld\n", prop.totalGlobalMem);
    printf("Shared memory size per block: %ld\n", prop.sharedMemPerBlock);
    printf("Number of registers per block: %d\n", prop.regsPerBlock);
    printf("Number of threads in a warp: %d\n", prop.warpSize);
    printf("Maximum number of threads that can be launched per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads that can be launched: %d x %d x %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum grid size: %d x %d x %d\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total constant memory: %ld\n", prop.totalConstMem);
    printf("Major compute capability: %d\n", prop.major);
    printf("Minor compute capability: %d\n", prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Maximum 1D texture memory: %d\n", prop.maxTexture1D);
    printf("Could we overlap? %d\n", prop.deviceOverlap);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Is there a limit for kernel execution? %d\n", prop.kernelExecTimeoutEnabled);
    printf("Is my GPU a chipsest? %d\n", prop.integrated);
    printf("Can we map the host memory? %d\n", prop.canMapHostMemory);
    printf("Can we launch concurrent kernels? %d\n", prop.concurrentKernels);
    printf("Do we have ECC memory? %d\n", prop.ECCEnabled);

    }

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
    }
