#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <curand.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <math.h>
#include "BlackandScholes.hpp"
#include "reduce.cuh"
#include "tool.cuh"
#include "option_price.hpp"
#include <curand_kernel.h>

#define CHECK_MALLOC(ptr) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "Memory allocation failed for %s at %s:%d\n", #ptr, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


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


__constant__ OptionData d_OptionData;

using namespace std;


void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in file " << file << " at line " << line
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Has to be defined in the compilation in order to get the correct value of the
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))


__global__ void setup_kernel(curandState *state, uint64_t seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}


__global__ void simulateOptionPriceGPU(float *d_optionPriceGPU, float K, float r, float T, float sigma, int N_PATHS,
                                       float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_PATHS) {
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = d_randomData[idx * N_STEPS + i];
            // cout << "G : " << G << endl;
            St *= exp((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
        }

        // // Calculate the payoff
        d_optionPriceGPU[idx] = max(St - K, 0.0f);


    }
}


// __global__ void
// simulateOptionPriceOneBlockGPUSumReduce(float *d_optionPriceGPU, float K, float r, float T, float sigma, int N_PATHS,
//                                         float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt,
//                                         float *output) {
//     int stride = blockDim.x;
//     int idx = threadIdx.x;
//     int tid = threadIdx.x;

//     // Shared memory for the block
//     __shared__ float sdata[1024];
//     float sum = 0.0f;

//     if (idx < N_PATHS) {
//         sdata[tid] = 0.0f;

//         while (idx < N_PATHS) {
//             float St = S0;
//             float G;
//             for (int i = 0; i < N_STEPS; i++) {
//                 G = d_randomData[idx * N_STEPS + i];
//                 // cout << "G : " << G << endl;
//                 St *= exp((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
//             }
//             sum += max(St - K, 0.0f);
//             idx += stride;
//         }
//         // Load input into shared memory
//         sdata[tid] = (tid < N_PATHS) ? sum : 0;

//         __syncthreads();

//         // Perform reduction in shared memory
//         for (unsigned int s = 1024 / 2; s > 0; s >>= 1) {
//             if (tid < s && (tid + s) < N_PATHS) {
//                 sdata[tid] += sdata[tid + s];
//             }
//             __syncthreads();
//         }

//         // Write result for this block to output
//         if (tid == 0) {
//             output[0] = sdata[0] * expf(-r);
//         }
//     }
// }
__global__ void
simulateOptionPriceMultipleBlockGPU(float *d_simulated_payoff, float K, float r, float T, float sigma, int N_PATHS,
                                    float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_PATHS) {
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = d_randomData[idx * N_STEPS + i];
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
        }
        d_simulated_payoff[idx] = max(St - K, 0.0f);
    }
}

__global__ void simulateOptionPriceMultipleBlockGPUwithReduce(float *g_odata, curandState *globalStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float T = d_OptionData.T;
    float sigma = d_OptionData.v;
    float S0 = d_OptionData.S0;
    int N_PATHS = d_OptionData.N_PATHS;
    float sqrdt = sqrtf(T);

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];


    if (idx < N_PATHS) {
        curandState state = globalStates[idx];
        float St = S0;
        float G, mySum;
        G = curand_normal(&state);
        St *= expf((r - sigma * sigma * 0.5) * T + sigma * sqrdt * G);
        mySum = max(St - K, 0.0f);
        sdata[tid] = mySum;
        cg::sync(cta);

        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
    }
}


__global__ void
simulateBulletOptionPriceMultipleBlockGPU(float *g_odata, curandState *globalStates, int Ik = 0, float Sk = 0.0f,
                                          int Tk = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float S0 = d_OptionData.S0;
    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float sigma = d_OptionData.v;
    float B = d_OptionData.B;
    int P1 = d_OptionData.P1;
    int P2 = d_OptionData.P2;
    int N_PATHS = d_OptionData.N_PATHS;
    int N_STEPS = d_OptionData.N_STEPS;
    float dt = d_OptionData.step;
    float sqrdt = sqrtf(dt);

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];


    if (idx < N_PATHS) {
        curandState state = globalStates[idx];
        int count = Ik;
        float St = (Sk == 0.0f) ? S0 : Sk;
        float G;
        int remaining_steps = N_STEPS - Tk;
        for (int i = 0; i < remaining_steps; i++) {
            G = curand_normal(&state);
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
        }
        if ((count >= P1) && (count <= P2)) {
            sdata[tid] = max(St - K, 0.0f);
        } else {
            sdata[tid] = 0.0f;
        }
        float mySum = sdata[tid];
        cg::sync(cta);

        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;

    }
}

__global__ void
simulateBulletOptionPriceMultipleBlockGPUatomic(float *g_odata, curandState *globalStates, int Ik = 0, float Sk = 0.0f,
                                                int Tk = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float S0 = d_OptionData.S0;
    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float sigma = d_OptionData.v;
    float B = d_OptionData.B;
    int P1 = d_OptionData.P1;
    int P2 = d_OptionData.P2;
    int N_PATHS = d_OptionData.N_PATHS;
    int N_STEPS = d_OptionData.N_STEPS;
    float dt = d_OptionData.step;
    float sqrdt = sqrtf(dt);

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];


    if (idx < N_PATHS) {
        curandState state = globalStates[idx];
        int count = Ik;
        float St = (Sk == 0.0f) ? S0 : Sk;
        float G;
        int remaining_steps = N_STEPS - Tk;
        for (int i = 0; i < remaining_steps; i++) {
            G = curand_normal(&state);
            St *= __expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
        }
        if ((count >= P1) && (count <= P2)) {
            sdata[tid] = max(St - K, 0.0f);
        } else {
            sdata[tid] = 0.0f;
        }
        float mySum = sdata[tid];
        cg::sync(cta);

        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) {
            atomicAdd(&(g_odata[0]), mySum);
        }

    }
}

__global__ void
simulate_outer_trajectories(float *g_odata, curandState *globalStates, float *d_stock_prices, int *d_sums_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float S0 = d_OptionData.S0;
    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float sigma = d_OptionData.v;
    float B = d_OptionData.B;
    int P1 = d_OptionData.P1;
    int P2 = d_OptionData.P2;
    int N_PATHS = d_OptionData.N_PATHS;
    int N_STEPS = d_OptionData.N_STEPS;
    float dt = d_OptionData.step;
    float sqrdt = sqrtf(dt);

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];


    if (idx < N_PATHS) {
        curandState state = globalStates[idx];
        int count = 0;
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = curand_normal(&state);
            St *= __expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
            d_sums_i[idx * N_STEPS + i] = count;
            d_stock_prices[idx * N_STEPS + i] = St;
        }
        if ((count >= P1) && (count <= P2)) {
            sdata[tid] = max(St - K, 0.0f);
        } else {
            sdata[tid] = 0.0f;
        }
        float mySum = sdata[tid];
        cg::sync(cta);

        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) {
            atomicAdd(&(g_odata[N_PATHS * N_STEPS]), mySum);
        }

    }
}


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
    int N_STEPS = option_data.N_STEPS;


    for (int i = 0; i < N_PATHS; i++) {
        float St = S0;
        for (int j = 0; j < N_STEPS; j++) {
            G = h_randomData[i * N_STEPS + j];
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);

        }

        countt += max(St - K, 0.0f);
    }
    *optionPriceCPU = expf(-r) * (countt / N_PATHS);
}


__global__ void
compute_nmc_one_block_per_point(float *d_option_prices, curandState *d_states, float *d_stock_prices, int *d_sums_i) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float sigma = d_OptionData.v;
    float B = d_OptionData.B;
    int P1 = d_OptionData.P1;
    int P2 = d_OptionData.P2;
    int N_PATHS = d_OptionData.N_PATHS;
    int N_PATHS_INNER = d_OptionData.N_PATHS_INNER;
    int N_STEPS = d_OptionData.N_STEPS;
    float dt = d_OptionData.step;
    float sqrdt = sqrtf(dt);

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];

    long unsigned int number_of_simulations = N_PATHS * N_STEPS;
    int number_of_blocks = gridDim.x;
    curandState state = d_states[idx];

    int count;
    float St;
    float G;
    int remaining_steps;
    tid = threadIdx.x;
    int tid_sim;
    while (blockId < number_of_simulations) {
        remaining_steps = N_STEPS - ((blockId % N_STEPS) + 1);
        float mySum = 0.0f;
        tid_sim = tid;
        while (tid_sim < N_PATHS_INNER) {

            count = d_sums_i[blockId];
            St = d_stock_prices[blockId];
            for (int i = 0; i < remaining_steps; i++) {
                G = curand_normal(&state);
                St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
                if (B > St) count += 1;
            }
            if ((count >= P1) && (count <= P2)) {
                mySum += max(St - K, 0.0f);


            } else {
                mySum += 0.0f;
            }
            tid_sim += blockSize;
        }
        sdata[tid] = mySum;
        cg::sync(cta);
        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) {
            //atomic add
            mySum = mySum * expf(-r) / static_cast<float>(N_PATHS_INNER);
            atomicAdd(&(d_option_prices[blockId]), mySum);

        }
        // if (cta.thread_rank() == 0 && blockId < number_of_simulations && blockId > (number_of_simulations - 100))  printf("blockId : %d, d_option_prices[blockId] : %f, d_sums_i[blockId] : %d, d_stock_prices[blockId] : %f, remaining_steps : %d\n", blockId, d_option_prices[blockId], d_sums_i[blockId], d_stock_prices[blockId], remaining_steps); 

        blockId += number_of_blocks;
    }


}


float wrapper_cpu_option_vanilla(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;

    float *d_randomData, *h_randomData;
    testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));
    h_randomData = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));
    generateRandomArray(d_randomData, h_randomData, N_PATHS, N_STEPS);


    float optionPriceCPU = 0.0f;
    simulateOptionPriceCPU(&optionPriceCPU, h_randomData, option_data);

    cout << endl;
    cout << "Average CPU : " << optionPriceCPU << endl << endl;
    free(h_randomData);
    cudaFree(d_randomData);

    return optionPriceCPU;
}

float wrapper_gpu_option_vanilla(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    // generate states array for random number generation
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, blocksPerGrid * sizeof(float)));
    float *h_odata = (float *) malloc(blocksPerGrid * sizeof(float));

    simulateOptionPriceMultipleBlockGPUwithReduce<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_odata[i];
    }
    float optionPriceGPU = expf(-option_data.r * option_data.T) * sum / N_PATHS;
    cout << "Average GPU : " << optionPriceGPU << endl << endl;
    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;
}

float wrapper_gpu_bullet_option(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, blocksPerGrid * sizeof(float)));
    float *h_odata = (float *) malloc(blocksPerGrid * sizeof(float));
    CHECK_MALLOC(h_odata);

    simulateBulletOptionPriceMultipleBlockGPU<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_odata[i];
    }
    float optionPriceGPU = expf(-option_data.r * option_data.T) * sum / static_cast<float>(N_PATHS);
    cout << "Average GPU bullet option : " << optionPriceGPU << endl << endl;

    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;

}

float wrapper_gpu_bullet_option_atomic(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, sizeof(float)));
    float *h_odata = (float *) malloc(sizeof(float));
    CHECK_MALLOC(h_odata);

    simulateBulletOptionPriceMultipleBlockGPUatomic<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

    float optionPriceGPU = expf(-option_data.r * option_data.T) * h_odata[0] / static_cast<float>(N_PATHS);
    cout << "Average GPU bullet option atomic : " << optionPriceGPU << endl << endl;

    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;

}


float
wrapper_gpu_bullet_option_nmc_one_point_one_block(OptionData option_data, int threadsPerBlock, int number_of_blocks) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states_outter, *d_states_inner;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);


    testCUDA(cudaMalloc(&d_states_outter, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states_outter, 1234);


    simulate_outer_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_option_prices, d_states_outter, d_stock_prices,
                                                                    d_sums_i);
    // testCUDA(cudaGetLastError());

    cudaDeviceSynchronize();
    cudaFree(d_states_outter);


    testCUDA(cudaMalloc(&d_states_inner, number_of_blocks * threadsPerBlock * sizeof(curandState)));

    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states_inner, 1235);
    // testCUDA(cudaGetLastError());


    size_t freeMem2;
    size_t totalMem2;
    testCUDA(cudaMemGetInfo(&freeMem2, &totalMem2));


    std::cout << "Free memory : " << freeMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Total memory : " << totalMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Used memory : " << (totalMem2 - freeMem2) / 1024 / 1024 << " MB\n";


    compute_nmc_one_block_per_point<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states_inner,
                                                                           d_stock_prices, d_sums_i);
    // testCUDA(cudaGetLastError());


    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, number_of_options * sizeof(int), cudaMemcpyDeviceToHost));

    //compute average of option prices
    float sum = 0.0f;
    for (int i = 0; i < number_of_options; i++) {
        sum += h_option_prices[i];
    }
    float callResult = sum / static_cast<float>(number_of_options);
    cout << "Average GPU bullet option nmc one point per block : " << callResult
         << endl << endl;


    free(h_option_prices);
    free(h_stock_prices);
    free(h_sums_i);
    cudaFree(d_option_prices);
    cudaFree(d_stock_prices);
    cudaFree(d_sums_i);

    return callResult;

}

float
get_max_number_of_blocks(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states_outter, *d_states_inner;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);


    testCUDA(cudaMalloc(&d_states_outter, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states_outter, 1234);


    simulate_outer_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_option_prices, d_states_outter, d_stock_prices,
                                                                    d_sums_i);
    testCUDA(cudaGetLastError());

    cudaDeviceSynchronize();
    cudaFree(d_states_outter);

    size_t freeMem;
    size_t totalMem;
    testCUDA(cudaMemGetInfo(&freeMem, &totalMem));

    int number_of_blocks = 10000;
    cudaError_t status;
    while (true) {
        status = cudaMalloc(&d_states_inner, number_of_blocks * threadsPerBlock * sizeof(curandState));

        if (status == cudaSuccess) {
            // Allocation successful, free memory and try a larger size
            cudaFree(d_states_inner);
            d_states_inner = nullptr;
            number_of_blocks += 10000;
        } else {
            cudaFree(d_states_inner);
            break;
        }
    }
    number_of_blocks *= 0.9f;
    cout << "max number of blocks : " << number_of_blocks << endl;

    cudaDeviceReset();
    return number_of_blocks;
}

__global__ void
compute_nmc_one_block_per_point_with_outter(float *d_option_prices, curandState *d_states, float *d_stock_prices,
                                            int *d_sums_i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int number_of_blocks = gridDim.x;

    float S0 = d_OptionData.S0;
    float K = d_OptionData.K;
    float r = d_OptionData.r;
    float sigma = d_OptionData.v;
    float B = d_OptionData.B;
    int P1 = d_OptionData.P1;
    int P2 = d_OptionData.P2;
    int N_PATHS = d_OptionData.N_PATHS;
    int N_STEPS = d_OptionData.N_STEPS;
    float dt = d_OptionData.step;
    float sqrdt = sqrtf(dt);
    int N_PATHS_INNER = d_OptionData.N_PATHS_INNER;

    int number_of_simulation_per_block = (N_PATHS + number_of_blocks - 1) / number_of_blocks;

    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];
    curandState state = d_states[idx];


    if (tid < number_of_simulation_per_block && (tid * number_of_blocks + blockIdx.x) < N_PATHS) {

        int count = 0;
        float St = S0;
        float G;
        int index;
        for (int i = 0; i < N_STEPS; i++) {
            G = curand_normal(&state);
            index = (tid * number_of_blocks + blockIdx.x) * N_STEPS + i;
            St *= __expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
            d_sums_i[index] = count;
            d_stock_prices[index] = St;
        }
        if ((count >= P1) && (count <= P2)) {
            sdata[tid] = max(St - K, 0.0f);
        } else {
            sdata[tid] = 0.0f;
        }
        float mySum = sdata[tid];
        cg::sync(cta);

        if ((blockSize >= 1024) && (tid < 512)) {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
        if ((blockSize >= 512) && (tid < 256)) {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128)) {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64)) {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32) {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) {
            atomicAdd(&(d_option_prices[N_PATHS * N_STEPS]), mySum);
        }
    }

    int compteur = 0;
    int remaining_steps;
    int tid_sim;
    float St;
    float G;
    int count;
    int blockId;

    while (compteur < number_of_simulation_per_block && (compteur * number_of_blocks + blockIdx.x) < N_PATHS) {

        for (int i = 0; i < N_STEPS; i++) {
            blockId = (compteur * number_of_blocks + blockIdx.x) * N_STEPS + i;
            // if (tid == 0 && blockId >= N_PATHS * N_STEPS ) {
            //     printf("We went too far : blockId : %d\n", blockId);
            // }
            remaining_steps = N_STEPS - ((blockId % N_STEPS) + 1);
            float mySum = 0.0f;
            tid_sim = tid;
            while (tid_sim < N_PATHS_INNER) {

                St = d_stock_prices[blockId];
                for (int i = 0; i < remaining_steps; i++) {
                    G = curand_normal(&state);
                    St *= __expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
                    if (B > St) count += 1;
                }
                if ((count >= P1) && (count <= P2)) {
                    mySum += max(St - K, 0.0f);


                } else {
                    mySum += 0.0f;
                }
                tid_sim += blockSize;
            }
            if(tid == 0) printf("mySum : %f, blockid : %d\n", mySum, blockId);
            sdata[tid] = mySum;
            cg::sync(cta);
            if ((blockSize >= 1024) && (tid < 512)) {
                sdata[tid] = mySum = mySum + sdata[tid + 512];
            }
            cg::sync(cta);
            if ((blockSize >= 512) && (tid < 256)) {
                sdata[tid] = mySum = mySum + sdata[tid + 256];
            }

            cg::sync(cta);

            if ((blockSize >= 256) && (tid < 128)) {
                sdata[tid] = mySum = mySum + sdata[tid + 128];
            }

            cg::sync(cta);

            if ((blockSize >= 128) && (tid < 64)) {
                sdata[tid] = mySum = mySum + sdata[tid + 64];
            }
            cg::sync(cta);


            cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

            if (cta.thread_rank() < 32) {
                // Fetch final intermediate sum from 2nd warp
                if (blockSize >= 64) mySum += sdata[tid + 32];
                // Reduce final warp using shuffle
                for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
                    mySum += tile32.shfl_down(mySum, offset);
                }
            }


            // write result for this block to global mem
            if (cta.thread_rank() == 0) {
                //atomic add
                mySum = mySum * expf(-r) / static_cast<float>(N_PATHS_INNER);
                atomicAdd(&(d_option_prices[blockId]), mySum);
            }
        }
        compteur += 1;
    }


}

float wrapper_gpu_bullet_option_nmc_one_kernel(OptionData option_data, int threadsPerBlock, int number_of_blocks) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);

    testCUDA(cudaMalloc(&d_states, number_of_blocks * threadsPerBlock * sizeof(curandState)));
    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states, 1234);
    testCUDA(cudaGetLastError());

    compute_nmc_one_block_per_point_with_outter<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states,
                                                                                       d_stock_prices, d_sums_i);
    cudaDeviceSynchronize();

    testCUDA(cudaGetLastError());

    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, number_of_options * sizeof(int), cudaMemcpyDeviceToHost));

    cout << "h_option_prices[N_PATHS * N_STEPS] : "
         << h_option_prices[N_PATHS * N_STEPS] * expf(-option_data.r * option_data.T) / static_cast<float>(N_PATHS)
         << endl;
    float sum = 0.0f;
    for (int i = 0; i < N_PATHS * N_STEPS; i++) {
        sum += h_option_prices[i];
    }

    for(int i = 0; i < 100; i++) {
        cout << "h_option_prices[" << i << "] : " << h_option_prices[i] << endl;
    }

    float callResult = sum / static_cast<float>(number_of_options);
    cout << "Average GPU bullet option nmc one kernel : " << callResult
         << endl << endl;


    free(h_option_prices);
    free(h_stock_prices);
    free(h_sums_i);
    cudaFree(d_option_prices);
    cudaFree(d_stock_prices);
    cudaFree(d_sums_i);

    return 0.0f;

}


int main(void) {

    OptionData option_data;
    option_data.S0 = 100.0f;
    option_data.T = 1.0f;
    option_data.K = 100.0f;
    option_data.r = 0.1f;
    option_data.v = 0.2f;
    option_data.B = 120.0f;
    option_data.P1 = 10;
    option_data.P2 = 50;
    option_data.N_PATHS = 10;
    option_data.N_PATHS_INNER = 5000;
    option_data.N_STEPS = 100;
    option_data.step = option_data.T / static_cast<float>(option_data.N_STEPS);

    int threadsPerBlock = 1024;

    // Copy option data to constant memory
    cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    printOptionData(option_data);

    // getDeviceProperty();

    // wrapper_cpu_option_vanilla(option_data, threadsPerBlock);

    // wrapper_gpu_option_vanilla(option_data, threadsPerBlock);

    // wrapper_gpu_bullet_option(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);
    // int max_number_of_block_to_everfow = get_max_number_of_blocks(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, 250000);


    wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, 20000);


    float callResult = 0.0f;
    black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    cout << endl << "call Black Scholes : " << callResult << endl;


    return 0;
}


