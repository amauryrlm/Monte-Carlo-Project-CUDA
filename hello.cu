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
simulateBulletOptionPriceMultipleBlockGPU(float *g_odata, curandState *globalStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float S0 = d_OptionData.S0;
    float T = d_OptionData.T;
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
simulateBulletOptionSavePrice(float *d_simulated_paths, float *d_simulated_count, float K, float r, float T,
                              float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt,
                              float sqrdt, float B, float P1, float P2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_PATHS) {
        int count = 0;
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = d_randomData[idx * N_STEPS + i];
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            if (B > St) count += 1;
            if ((count >= P1) && (count <= P2)) {
                d_simulated_paths[idx * N_STEPS + i] = St;
            } else {
                d_simulated_paths[idx * N_STEPS + i] = 0.0f;
            }
            d_simulated_count[idx * N_STEPS + i] = count;
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
    cout << endl << "number generated" << endl;
    testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "host copied" << endl;
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

float wrapper_cpu_option_vanilla(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;


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
    cout << "blocksPerGrid bullet : " << blocksPerGrid << endl;
    // generate states array for random number generation
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
    option_data.N_PATHS = 100000;
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


    float callResult = 0.0f;
    black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    cout << endl << "call Black Scholes : " << callResult << endl;






// //-------------------------------BULLET OPTION WITH MULTIPLE BLOCKS ------------------------------------------------------------
// //------------------------------------------------------------------------------------------------------------------------------

//     float *d_simulated_payoff_bullet, *h_simulated_payoff_bullet, *h_output4, *d_output4;
//     testCUDA(cudaMalloc((void **) &d_simulated_payoff_bullet, N_PATHS * sizeof(float)));
//     testCUDA(cudaMalloc((void **) &d_output4, blocks * sizeof(float)));
//     h_output4 = (float *) malloc(blocks * sizeof(float));
//     h_simulated_payoff_bullet = (float *) malloc(N_PATHS * sizeof(float));

//     simulateBulletOptionPriceMultipleBlockGPU<<<blocksPerGrid, threadsPerBlock>>>(d_simulated_payoff_bullet, K, r, T,
//                                                                                   sigma, N_PATHS, d_randomData, N_STEPS,
//                                                                                   S0, dt, sqrdt, B, P1, P2);
//     cudaError_t error4 = cudaGetLastError();
//     if (error4 != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error4));
//         return -1;
//     }
//     cudaDeviceSynchronize();
//     testCUDA(cudaMemcpy(h_simulated_payoff_bullet, d_simulated_payoff_bullet, N_PATHS * sizeof(float),
//                         cudaMemcpyDeviceToHost));

//     reduce3<<<blocks, threads>>>(d_simulated_payoff_bullet, d_output4, N_PATHS);
//     error4 = cudaGetLastError();
//     if (error4 != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error4));
//         return -1;
//     }
//     cudaDeviceSynchronize();
//     testCUDA(cudaMemcpy(h_output4, d_output4, blocks * sizeof(float), cudaMemcpyDeviceToHost));
//     float sum4 = 0.0f;
//     for (int i = 0; i < blocks; i++) {
//         sum4 += h_output4[i];
//     }
//     cout << "result gpu cuda computed bullet option " << expf(-r * T) * sum4 / N_PATHS << endl;



//     //-------------------------------BULLET OPTION WITH MULTIPLE BLOCKS AND SAVE PATHS------------------------------------------------------------
//     //-------------------------------------------------------------------------------------------------------------------------------------------

//     float *d_simulated_paths, *d_simulated_count, *h_simulated_paths, *h_simulated_count;
//     testCUDA(cudaMalloc((void **) &d_simulated_paths, N_PATHS * N_STEPS * sizeof(float)));
//     testCUDA(cudaMalloc((void **) &d_simulated_count, N_PATHS * N_STEPS * sizeof(float)));
//     h_simulated_paths = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));
//     h_simulated_count = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));

//     simulateBulletOptionSavePrice<<<blocksPerGrid, threadsPerBlock>>>(d_simulated_paths, d_simulated_count, K, r, T,
//                                                                       sigma, N_PATHS, d_randomData, N_STEPS, S0, dt,
//                                                                       sqrdt, B, P1, P2);
//     cudaError_t error5 = cudaGetLastError();
//     if (error5 != cudaSuccess) {
//         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error5));
//         return -1;
//     }
//     cudaDeviceSynchronize();
//     testCUDA(cudaMemcpy(h_simulated_paths, d_simulated_paths, N_PATHS * N_STEPS * sizeof(float),
//                         cudaMemcpyDeviceToHost));
//     testCUDA(cudaMemcpy(h_simulated_count, d_simulated_count, N_PATHS * N_STEPS * sizeof(float),
//                         cudaMemcpyDeviceToHost));


//     cudaFree(d_simulated_paths);
//     cudaFree(d_simulated_count);
//     free(h_simulated_paths);
//     free(h_simulated_count);
//     cudaFree(d_simulated_payoff_bullet);
//     cudaFree(d_output4);
//     free(h_output4);
//     cudaFree(d_randomData);

//     //close file
//     fclose(file);


    return 0;
}


