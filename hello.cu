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


using namespace std;

// template<class Sumarize>

__global__ void myKernel(void) {
}
int main(void) {

// declare variables and constants
    const size_t N_PATHS = 10;
    const size_t N_STEPS = 5;
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
            // cout << "G : " << G << endl;
            St *= exp((r - (sigma*sigma)/2)*step + sigma * sqrt(step) * G);
            cout << "St : " << St << endl;
        }
        cout << "S before assigning " << St << endl;
        s[i] = St;
        cout << "S " << St << endl;
        cout << i << endl;
    }
    cout << "paths calculated";
    // generate random numbers using curand

    //allocate array filled with random values 
    float *d_randomData;
    testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));

    // // create generator all fill array with generated values
    // curandGenerator_t gen;
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // curandGenerateNormal(gen, d_randomData, N_PATHS * N_STEPS, 0.0, 1.0);


    // float h_randomData[N_PATHS * N_STEPS];
    // cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N_PATHS * N_STEPS; i++) {
    //     cout << "random  : " << d_randomData[i] << endl;
    // }


    // cudaFree(d_randomData);
    // curandDestroyGenerator(gen);

	return 0;
}

