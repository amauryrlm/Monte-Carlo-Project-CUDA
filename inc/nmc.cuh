#pragma once
// Implementation of Nested Monte Carlo functions
#include <curand.h>
#include <curand_kernel.h>

#include "reduce.cuh"
#include "tool.cuh"
#include "trajectories.cuh"

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
    float mySum;
    while (compteur < number_of_simulation_per_block && (compteur * number_of_blocks + blockIdx.x) < N_PATHS) {

        for (int i = 0; i < N_STEPS; i++) {
            blockId = (compteur * number_of_blocks + blockIdx.x) * N_STEPS + i;

            remaining_steps = N_STEPS - ((blockId % N_STEPS) + 1);
            
            tid_sim = tid;
            mySum = 0.0f;
            while (tid_sim < N_PATHS_INNER) {
                count = d_sums_i[blockId];
                St = d_stock_prices[blockId];
                for (int i = 0; i < remaining_steps; i++) {
                    G = curand_normal(&state);
                    St *= __expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
                    if (B > St) count += 1;
                }
                if ((count >= P1) && (count <= P2)) {
                    mySum += max(St - K, 0.0f);

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
            if (cta.thread_rank() == 0 ) {
                //atomic add
                mySum = mySum * __expf(-r) / static_cast<float>(N_PATHS_INNER);
                atomicAdd(&(d_option_prices[blockId]), mySum);
            }
        }
        compteur += 1;
    }


}