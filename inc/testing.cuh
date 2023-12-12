// Set of tools to test our MC implementations

#pragma once


// #include "amaury.cuh"
#include <cmath>
#include <iostream>
#include <vector>

#include "tool.cuh"
#include "reduce.cuh"
#include "trajectories.cuh"
#include "nmc.cuh"


void generate_random_array(float *d_randomData, float *h_randomData, int length, unsigned long long seed = 1234ULL) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_randomData, length, 0.0, 1.0);
    testCUDA(cudaMemcpy(h_randomData, d_randomData, length * sizeof(float), cudaMemcpyDeviceToHost));
    curandDestroyGenerator(gen);
}

/**
 * @brief Allocate space for and fill the elements of an array with size `length` on both the CPU and GPU.
 *
 * @param d_randomData
 * @param h_randomData
 * @param length
 * @param seed
 */
void init_random_array(float **d_randomData, float **h_randomData, size_t length, long seed = 1234ULL) {

    size_t n_random_bytes = length * sizeof(float);

    testCUDA(cudaMalloc(d_randomData, n_random_bytes));
    *h_randomData = (float *) malloc(n_random_bytes);
    generate_random_array(*d_randomData, *h_randomData, length, seed);

}



__global__ void simulateOptionPriceMultipleBlockGPU(
    float *d_simulated_payoff,
    float *d_simulated_trajectories,
    curandState *globalStates,
    float K,
    float r,
    float T,
    float sigma,
    int N_PATHS,
    int N_STEPS,
    float S0,
    float dt,
    float sqrdt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step_idx = idx * N_STEPS;

    if (idx < N_PATHS) {
        float St = S0;
        float G;
        for (int i = 0; i < N_STEPS; i++) {
            G = curand_normal(&globalStates[idx]);
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);
            d_simulated_trajectories[step_idx + i] = St;
        }
        d_simulated_payoff[idx] = max(St - K, 0.0f);
    }
}

void simulateOptionPriceCPU(float *optionPriceCPU, int N_PATHS, int N_STEPS, float * h_randomData, float S0, float sigma, float sqrdt, float r, float K, float dt, float *simulated_paths_cpu){
    float G;
    float countt = 0.0f;
    for(int i=0; i<N_PATHS;i++){
        float St = S0;
        for(int j=0; j<N_STEPS; j++){
            G = h_randomData[i*N_STEPS + j];
            St *= expf((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);

        }

        simulated_paths_cpu[i] = max(St - K, 0.0f);
        // cout << "cpu : " <<  St << endl;
        countt += max(St - K, 0.0f);
    }
    *optionPriceCPU = countt/N_PATHS;
}



__global__ void init_rng_kernel(curandState *state, uint64_t seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

enum ReductionType {
    SequentialAddressing = 3,
    FirstAddDuringLoad = 4,
    UnrollLastWarp = 5,
    CompletelyUnrolled = 6
    // MultipleAddsPerThread = 7
};

class Simulation {

public:

    size_t n_trajectories;
    size_t n_steps;
    float * d_random_array = nullptr;
    float * h_random_array = nullptr;
    curandState *d_states = nullptr;

    Simulation (
        size_t n_trajectories = 10,
        size_t n_steps = 100,
        float volatilty = 0.2,
        float risk_free_rate = 0.1,
        float initial_spot_price = 100.0,
        float contract_strike = 100.0,
        float contract_maturity = 1,
        float barrier = 0,
        float P1 = 0,
        float P2 = 0
    )
        : n_trajectories{n_trajectories}
        , n_steps{n_steps}
        , sigma{volatilty}
        , r{risk_free_rate}
        , x_0{initial_spot_price}
        , K{contract_strike}
        , T{contract_maturity}
        , B{barrier}
        , P1{P1}
        , P2{P2}
    {

        this->initialize_random_array();
        // this->initialize_rng_state();

    }

    void initialize_rng_state(size_t threads_per_block, uint64_t seed = 1234) {
        int blocks_per_grid = (this->n_trajectories + threads_per_block - 1) / threads_per_block;

        testCUDA(cudaMalloc(&this->d_states, this->n_trajectories * sizeof(curandState)));
        init_rng_kernel<<<blocks_per_grid, threads_per_block>>>(this->d_states, seed);

    }


    /**
     * @brief Compute the sum of the contents stored in Simulation.h_random_array.
     *
     * @return float
     */
    float sum_random_array() {

        if (!this->h_random_array) {
            this->initialize_random_array();
        }

        float out = 0.0;
        int n = this->length();
        for (int i = 0; i < n; i++) {
            out += this->h_random_array[i];
        }

        return out;
    }


    /**
     * @brief Perform a test reduction, returning a vector storing the result of each block.
     *
     * @param n_blocks
     * @param n_threads_per_block
     * @param reduction
     * @return std::vector<float>
     */
    std::vector<float> test_reduction(size_t n_blocks, size_t n_threads_per_block, int reduction) {

        // Is our random array initialized?
        if (!this->d_random_array) {
            this->initialize_random_array();
        }

        float * g_idata = this->d_random_array;
        float * g_odata;

        // Allocate space for the output data
        testCUDA(cudaMalloc(&g_odata, n_blocks * sizeof(float)));
        size_t n = this->length();

        // Now compute the value of the reduction

        switch (reduction) {

            case SequentialAddressing: // 3

                reduce3<<<n_blocks, n_threads_per_block>>>(g_idata, g_odata, n);
                break;

            case FirstAddDuringLoad: // 4

                reduce4<<<n_blocks, n_threads_per_block>>>(g_idata, g_odata, n);
                break;

            case UnrollLastWarp: // 5

                reduce5<<<n_blocks, n_threads_per_block>>>(g_idata, g_odata, n);
                break;

            case CompletelyUnrolled: // 6

                // Is our length a power of 2 ?
                bool is_power_2 = isPow2(n);
                reduce6<<<n_blocks, n_threads_per_block>>>(g_idata, g_odata, n, is_power_2);
                break;

        }

        // Move bytes from g_odata into a vector on the cpu.
        std::vector<float> out(n_blocks);

        testCUDA(cudaMemcpy(out.data(), g_odata, n_blocks * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(g_odata);

        return out;
    }


    OptionData to_option_data() {

        OptionData option_data {
            .S0 = this->x_0,
            .T = this->T,
            .K = this->K,
            .r = this->r,
            .v = this->sigma,
            .B = this->B ,
            .P1 = this->P1,
            .P2 = this->P2,
            .N_PATHS = this->n_trajectories,
            .N_STEPS = this->n_steps
        };

        return option_data;
    }


    /* -------------------------- Simulation functions -------------------------- */
    std::vector<float> simulate_trajectory_cpu() {

        // First check to see if we have random numbers already simulated.
        // Let's simulate different trajectories now.

        if(!this->d_random_array || !this->h_random_array) {
            std::cout << "Initializing random array with " << this->length() << " elements\n";
            this->initialize_random_array();
        }

        std::cout << "Simulating trajectories using reduction method: \n";

        // Allocate memomry for the output array.
        float option_price = 0.0;
        std::vector<float> simulated_paths_cpu(this->n_trajectories, 0.0);

        simulateOptionPriceCPU(
            &option_price,
            this->n_trajectories,
            this->n_steps,
            this->h_random_array,
            this->initial_spot_price(),
            this->volatility(),
            this->sqrt_dt(),
            this->risk_free_rate(),
            this->contract_strike(),
            this->dt(),
            simulated_paths_cpu.data()
        );

        // Now return the vector containing all of the simulations
        return simulated_paths_cpu;

    }

    /**
     * @brief Simulate the outer trajectories to be used in a NMC implementation.
     *
     * @param n_blocks
     * @param n_threads_per_block
     * @return std::vector<float>
     */
    std::vector<float> simulate_outer_trajectories(size_t n_threads_per_block, uint64_t seed) {

        int blocks_per_grid = (this->n_trajectories + n_threads_per_block - 1) / n_threads_per_block;
        // Allocate space for the rng.
        this->initialize_rng_state(n_threads_per_block, seed);

        std::cout << "====================================================================\n";
        std::cout << "Going to compute outer trajectories...\n";
        std::cout << "Number of threads per block: " << n_threads_per_block << "\n";
        std::cout << "Number of trajectories: " << this->n_trajectories << "\n";
        std::cout << "Number of blocks: " << blocks_per_grid << "\n";
        std::cout << "Number of steps: " << this->n_steps << "\n";
        std::cout << "====================================================================\n";

        // Allocate space for the trajectories
        float *d_payoffs;
        float *d_trajectories;
        cudaMalloc(&d_trajectories, sizeof(float) * this->length());
        cudaMalloc(&d_payoffs, sizeof(float) * this->n_trajectories);

        // Now simulate the trajectory
        simulateOptionPriceMultipleBlockGPU<<<blocks_per_grid, n_threads_per_block>>>(
            d_payoffs,
            d_trajectories,
            this->d_states,
            this->contract_strike(),
            this->risk_free_rate(),
            this->contract_maturity(),
            this->volatility(),
            this->n_trajectories,
            this->n_steps,
            this->initial_spot_price(),
            this->dt(),
            this->sqrt_dt()
        );

        // Now copy this back to the cpu and return the results as a vector.
        std::vector<float> out(this->length());

        cudaMemcpy(out.data(), d_trajectories, this->length() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_trajectories);
        cudaFree(d_payoffs);
        cudaFree(this->d_states);

        return out;
    }





    size_t length() {
        return this->n_steps * this->n_trajectories;
    }

    void initialize_random_array(size_t seed = 1234ULL) {

        // Check if the pointers are empty
        if (this->d_random_array) {
            testCUDA(cudaFree(this->d_random_array));
        }

        if (this->h_random_array) {
            free(this->h_random_array);
        }

        init_random_array(&(this->d_random_array), &(this->h_random_array), this->length(), seed);
    }

    float &volatility() {
        return this->sigma;
    }

    float &risk_free_rate() {
        return this->r;
    }

    /**
     * @brief Return the initial spot price of our simulation. Known as x0 or S0
     *
     * @return float
     */
    float &initial_spot_price() {
        return this->x_0;
    }

    /**
     * @brief Return the contract strike (K) of our simulation.
     *
     * @return float
     */
    float &contract_strike() {
        return this->K;
    }

    float &contract_maturity() {
        return this->T;
    }

    float &barrier() {
        return this->B;
    }

    float dt() {
        return this->contract_maturity() / this->n_steps;
    }

    float sqrt_dt() {
        return sqrt(this->dt());
    }


    float sigma;   // volatility
    float r;       // risk-free rate
    float x_0;     // initial_spot_price
    float K;       // contract_strike
    float T;       // contract maturity
    float B;       // barrier
    float P1;
    float P2;

// private:


};

// Now, from this class we want to launch different testing suites
