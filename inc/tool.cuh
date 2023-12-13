#pragma once

#include <math.h>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
using namespace std;

// Parameters for option pricing
struct OptionData {
    float S0;
    float T;
    float K;
    float r;
    float v;
    float B;
    int P1;
    int P2;
    int N_PATHS;
    int N_PATHS_INNER;
    int N_STEPS;
    float step;
};

void printOptionData(OptionData od) {
    cout << endl;
    cout << "S0 : " << od.S0 << endl;
    cout << "T : " << od.T << endl;
    cout << "K : " << od.K << endl;
    cout << "r : " << od.r << endl;
    cout << "v : " << od.v << endl;
    cout << "B : " << od.B << endl;
    cout << "P1 : " << od.P1 << endl;
    cout << "P2 : " << od.P2 << endl;
    cout << "N_PATHS : " << od.N_PATHS << endl;
    cout << "N_PATHS_INNER : " << od.N_PATHS_INNER << endl;
    cout << "N_STEPS : " << od.N_STEPS << endl;
    cout << "step : " << od.step << endl;
    cout << endl;
}


#define CHECK_MALLOC(ptr) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "Memory allocation failed for %s at %s:%d\n", #ptr, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void getDeviceProperty() {

    const float GIGA = 1024.0 * 1024.0 * 1024.0;

    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    printf("The number of devices available is %i GPUs \n", count);
    cudaGetDeviceProperties(&prop, count - 1);
    printf("Name: %s\n", prop.name);
    printf("Global memory size in bytes: %fGB\n", prop.totalGlobalMem / GIGA);
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


void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in file " << file << " at line " << line
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

void simulateOptionPriceCPU(float *optionPriceCPU, float *h_randomData, OptionData option_data) {
    float G;
    float countt = 0.0f;
    float K = option_data.K;
    float r = option_data.r;
    float sigma = option_data.v;
    float S0 = option_data.S0;
    float dt = option_data.step;
    float sqrdt = sqrtf(dt);
    int N_PATHS = option_data.N_PATHS;
    float T = option_data.T;


    for (int i = 0; i < N_PATHS; i++) {
        float St = S0;
        G = h_randomData[i];
        St *= expf((r - (sigma * sigma) / 2) * T + sigma * sqrdt * G);

        countt += max(St - K, 0.0f);
    }
    *optionPriceCPU = expf(-r * T) * countt / static_cast<float>(N_PATHS);
}
void simulateBulletOptionPriceCPU(float *optionPriceCPU, float *h_randomData, OptionData option_data) {
    float G;
    float countt = 0.0f;
    float K = option_data.K;
    float r = option_data.r;
    float sigma = option_data.v;
    float S0 = option_data.S0;
    float dt = option_data.step;
    float sqrdt = sqrtf(dt);
    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    float B = option_data.B;
    float P1 = option_data.P1;
    float P2 = option_data.P2;
    float T = option_data.T;
    int count;
    float St;


    for (int i = 0; i < N_PATHS; i++) {
        St = S0;
        count = 0;

        for (int j = 0; j < N_STEPS; j++) {
            G = h_randomData[i * N_STEPS + j];
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (St > B) {
                count++;
            }
        }
        if (count >= P1 && count <= P2) {
            countt += max(St - K, 0.0f);
        }
    }
    *optionPriceCPU = expf(-r * T) * countt / static_cast<float>(N_PATHS);
}

// Return the maximum number of blocks that we can run using global memory for RNG state.
size_t get_max_blocks(int threads_per_block) {

    // We also need to reserve space for n_trajectories and n_steps
    // Figure out how much memory there is.
    size_t free_mem;
    size_t total_mem;
    testCUDA(cudaMemGetInfo(&free_mem, &total_mem));
    printf("free_mem: %7.3fGB, total_mem: %7.3fGB\n", free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));

    // Use 90% of total memory:
    float multiplier = 0.90;
    return (size_t) (free_mem * multiplier) / (sizeof(curandState_t) * threads_per_block);
}

void generateRandomArray(float *d_randomData, float *h_randomData, int N_PATHS, int N_STEPS,
                         unsigned long long seed = 1234ULL) {

    // create generator all fill array with generated values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_randomData, N_PATHS * N_STEPS, 0.0, 1.0);
    testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
    curandDestroyGenerator(gen);

}


__global__ void setup_kernel(curandState *state, uint64_t seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
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
