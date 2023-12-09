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
    int N_STEPS;
    float step;
};


__constant__ OptionData d_OptionData;

using namespace std;


// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
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

//1024 traj per block each time until end
__global__ void
simulateBulletOptionOutter(float *d_option_prices, curandState *d_states, float *d_stock_prices, float *d_sums_i) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int maxnumbreprice = d_OptionData.N_PATHS * d_OptionData.N_STEPS;

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
        curandState state = d_states[idx];
        int count = 0;
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = curand_normal(&state);
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
            d_stock_prices[idx * N_STEPS + i] = St;
            d_sums_i[idx * N_STEPS + i] = count;
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
            printf("mySum : %f\n", mySum);
            //atomic add
            atomicAdd(&(d_option_prices[N_PATHS*N_STEPS]), mySum);
            printf("d_option_prices[N_PATHS*N_STEPS + 1 ] : %f\n", d_option_prices[N_PATHS * N_STEPS]);
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
compute_nmc_one_block_per_point(float *d_option_prices, curandState *d_states, float *d_stock_prices, float *d_sums_i) {
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

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

    long unsigned int number_of_simulations = N_PATHS * N_STEPS;
    int number_of_blocks = gridDim.x;

    while (blockId < number_of_simulations) {
        curandState state = d_states[blockId];
        int count = d_sums_i[blockId];
        float St = d_stock_prices[blockId];
        float G;
        tid = threadIdx.x;

        while (tid < N_PATHS) {
            for (int i = 0; i < N_STEPS; i++) {
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
            if (cta.thread_rank() == 0) {
                //atomic add
                atomicAdd(&(d_option_prices[blockId]), mySum);
            }
            tid += blockSize;
        }
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
    float optionPriceGPU = expf(-option_data.r * option_data.T) * sum / N_PATHS;
    cout << "Average GPU bullet option : " << optionPriceGPU << endl << endl;

    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;

}

void wrapper_gpu_bullet_option_nmc(OptionData option_data, int threadsPerBlock, int number_of_blocks) {
    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int maxnumbreprice = N_PATHS * N_STEPS +1;


    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, number_of_blocks * sizeof(curandState)));
    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states, 1234);

    float *d_option_prices, *d_stock_prices, *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, maxnumbreprice * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, N_PATHS * N_STEPS * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, N_PATHS * N_STEPS * sizeof(float)));

    float *h_option_prices = (float *) malloc(maxnumbreprice * sizeof(float));
    float *h_stock_prices = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));
    float *h_sums_i = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));

    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;

    simulateBulletOptionOutter<<<blocksPerGrid, threadsPerBlock>>>(d_option_prices, d_states, d_stock_prices,
                                                                      d_sums_i);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();

    // compute_nmc_one_block_per_point<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states, d_stock_prices,
    //                                                                        d_sums_i);
    // cudaError_t error2 = cudaGetLastError();
    // if (error2 != cudaSuccess) {
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error2));
    // }
    // cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, maxnumbreprice * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 1000; i < N_PATHS; i++) {
        for (int j = 0; j < N_STEPS; j++) {
            cout << "simulations : " << i << " steps : " << j << " stock price : " << h_stock_prices[i * N_STEPS + j]
                 << " sum : " << h_sums_i[i * N_STEPS + j] << " option price : " << h_option_prices[i * N_STEPS + j]
                 << endl;
        }
    }

    cout << "Average GPU bullet option nmc : " << h_option_prices[maxnumbreprice - 1 ] << endl << endl;


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
    option_data.N_PATHS = 1024;
    option_data.N_STEPS = 100;
    option_data.step = option_data.T / static_cast<float>(option_data.N_STEPS);

    int threadsPerBlock = 1024;

    // Copy option data to constant memory
    cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    printOptionData(option_data);

    getDeviceProperty();

    wrapper_cpu_option_vanilla(option_data, threadsPerBlock);

    wrapper_gpu_option_vanilla(option_data, threadsPerBlock);

    wrapper_gpu_bullet_option(option_data, threadsPerBlock);

    wrapper_gpu_bullet_option_nmc(option_data, threadsPerBlock, 1);


    float callResult = 0.0f;
    black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    cout << endl << "call Black Scholes : " << callResult << endl;


    return 0;
}


