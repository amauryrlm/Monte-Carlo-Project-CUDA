#include <iostream>
// #include <format>
// #include <functional>
#include <cuda_runtime.h>

#include "trajectories.hpp"
#include "common.hpp"
#include "Xoshiro.hpp"
#include  "pricinghost.hpp"
#include <random>
#include <curand.h>



using namespace std;

// template<class Sumarize>

__global__ void myKernel(void) {
}
int main(void) {

// declare variables and constants
    const size_t N_PATHS = 10;
    const size_t N_STEPS = 100;
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

    float step = 1.0f / N_STEPS;
    float G = 0.0f;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    cout << "step : " << step << endl;
    G = distribution(generator);
    // cout << "G : " << G;





    for(int i=0; i<N_PATHS;i++){
        float St = S0;
        for(int j=0; j<N_STEPS; j++){
            G = distribution(generator);
            cout << "G : " << G << endl;
            St *= exp((r - (sigma*sigma)/2)*step + sigma * sqrt(step) * G);
            cout << "St : " << St << endl;
        }
        s[i] = St;
    }

    // generate random numbers using curand

    //allocate array filled with random values 
    float *d_randomData;
    cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float));

    // create generator all fill array with generated values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_randomData, N, 0.0, 1.0);


    float h_randomData[N];
    cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N_PATHS * N_STEPS; i++) {
        cout << "random  : " << d_randomData[i] << endl;
    }


    cudaFree(d_randomData);
    curandDestroyGenerator(gen);

	return 0;
}

