#include <iostream>
// #include <format>
// #include <functional>
#include <cuda_runtime.h>

#include "trajectories.hpp"
#include "common.hpp"
#include "Xoshiro.hpp"
#include  "pricinghost.hpp"
#include <random>



using namespace std;

// template<class Sumarize>

__global__ void myKernel(void) {
}
int main(void) {

// declare variables and constants
    const size_t N_PATHS = 100000;
    const size_t N_STEPS = 365;
    const size_t N_NORMALS = N_PATHS*N_STEPS;
    const float T = 1.0f;
    const float K = 100.0f;
    const float B = 95.0f;
    const float S0 = 100.0f;
    const float sigma = 0.2f;
    const float mu = 0.1f;
    const float r = 0.05f;
    float dt = float(T)/float(N_STEPS);
    float sqrdt = sqrt(dt);

    vector<float> s(N_PATHS);

    float step = 1.0 / N_STEPS;
    float G = 0.0;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    cout << "step : " << step;
    G = distribution(generator);
    cout << "G : " << G;


    for(int i=0; i<N_PATHS;i++){
        float St = S0;
        for(int j=1; j<N_STEPS; j++){
            G = distribution(generator);
            // St *= exp((r - (sigma**2)/2)*step + sigma * sqrt(step) * G);
        }
    }

	return 0;
}

