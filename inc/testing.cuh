// Set of tools to test our MC implementations

#pragma once


#include "amaury.cuh"
#include <cmath>
#include <iostream>
#include <vector>


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

    Simulation (
        size_t n_trajectories = 10,
        size_t n_steps = 100,
        float volatilty = 0.2,
        float risk_free_rate = 0.1,
        float initial_spot_price = 100.0,
        float contract_strike = 100.0,
        float contract_maturity = 1,
        float barrier = 0
    )
        : n_trajectories{n_trajectories}
        , n_steps{n_steps}
        , sigma{volatilty}
        , r{risk_free_rate}
        , x_0{initial_spot_price}
        , K{contract_strike}
        , T{contract_maturity}
        , B{barrier}
    {

        this->initialize_random_array();

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

    float volatility() {
        return this->sigma;
    }

    float risk_free_rate() {
        return this->r;
    }

    /**
     * @brief Return the initial spot price of our simulation. Known as x0 or S0
     *
     * @return float
     */
    float initial_spot_price() {
        return this->x_0;
    }

    /**
     * @brief Return the contract strike (K) of our simulation.
     *
     * @return float
     */
    float contract_strike() {
        return this->K;
    }

    float contract_maturity() {
        return this->T;
    }

    float barrier() {
        return this->B;
    }

    float dt() {
        return this->contract_maturity() / this->n_steps;
    }

    float sqrt_dt() {
        return sqrt(this->dt());
    }



private:

    float sigma;   // volatility
    float r;       // risk-free rate
    float x_0;     // initial_spot_price
    float K;       // contract_strike
    float T;       // contract maturity
    float B;       // barrier

};

// Now, from this class we want to launch different testing suites