// Set of tools to test our MC implementations

#pragma once

#include "amaury.cuh"

/**
 * @brief Allocate space for and fill the elements of an array with size `length` on both the CPU and GPU.
 *
 * @param d_randomData
 * @param h_randomData
 * @param length
 * @param seed
 */
void initRandomArray(float **d_randomData, float **h_randomData, size_t length, long seed = 1234ULL) {

    size_t n_random_bytes = length * sizeof(float);

    testCUDA(cudaMalloc(d_randomData, n_random_bytes));
    *h_randomData = (float *) malloc(n_random_bytes);
    generate_random_array(*d_randomData, *h_randomData, length, seed);

}



class SimulationParameters {

public:

    size_t n_trajectories;
    size_t n_steps;
    float * d_random_array = nullptr;
    float * h_random_array = nullptr;

    SimulationParameters(
        size_t n_trajectories = 10,
        size_t n_steps = 100,
        float volatilty = 0.2,
        float risk_free_rate = 0.1,
        float initial_spot_price = 100.0,
        float contract_strike = 100.0,
        float contract_maturity = 1,
        float barrier = 0,
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

        this->init_random_array()

    }


    /* -------------------------- Simulation functions -------------------------- */
    void simulate_trajectory_gpu() {





    }

    size_t length() {
        return this->n_steps * this->n_trajectories;
    }

    void init_random_array(size_t seed = 1234ULL) {

        // Check if the pointers are empty
        if (!this->d_random_array) {
            testCUDA(cudaFree(this->d_random_array));
        }

        if (!this->h_random_array) {
            free(this->h_random_array);
        }

        initRandomArray(&(this->d_random_array), &(this->h_random_array), this->length(), seed);
    }

    float volatility() {
        return this->sigma;
    }

    float risk_free_rate() {
        return this->r;
    }

    float initial_spot_price() {
        return this->x_0;
    }

    float contract_strike() {
        return this->K;
    }

    float contract_maturity() {
        return this->T;
    }

    float barrier() {
        return this->B;
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