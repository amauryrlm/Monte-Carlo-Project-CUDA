#pragma once
// Device code to simulate MC and NMC option pricing
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "BlackandScholes.hpp"
#include "tool.cuh"
#include "reduce.cuh"

__constant__ OptionData d_OptionData;

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

__global__ void
simulateBulletOptionOutter(float *d_option_prices, curandState *globalStates, float *d_stock_prices, float *d_sums_i) {
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
            atomicAdd(&(d_option_prices[0]), mySum);
            printf("d_option_prices[0] : %f\n", d_option_prices[0]);
        }

    }


}

