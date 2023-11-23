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


__global__ void setValuesKernel(float *arr, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}




// Has to be defined in the compilation in order to get the correct value of the
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))


using namespace std;

__global__ void simulateOptionPrice(float *d_optionPriceGPU, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_PATHS) {
        // float St = S0;
        // float G;
        // for(int i = 0; i < N_STEPS; i++){
        //     G = d_randomData[idx*i];
        //     // cout << "G : " << G << endl;
        //     St *= exp((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
        // }
        
        // // Calculate the payoff
        d_optionPriceGPU[idx] = 1.0f;
    }
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






    // generate random numbers using curand

    //allocate array filled with random values 
    float *d_randomData;
    testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));

    // create generator all fill array with generated values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_randomData, N_PATHS * N_STEPS, 0.0, 1.0);

    cout << "number generated";


    float h_randomData[N_PATHS * N_STEPS];
    testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "host copied" << endl;
    cout << h_randomData[0];

    // for(int i = 0; i < N_PATHS * N_STEPS; i++) {
    //     cout << "random  : " << h_randomData[i] << endl;
    // }

    float count = 0.0f;
    for(int i=0; i<N_PATHS;i++){
        float St = S0;
        for(int j=0; j<N_STEPS; j++){
            G = h_randomData[i*j];
            // cout << "G : " << G << endl;
            St *= exp((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
            
        }
        // cout << "S before assigning " << St << endl;
        s[i] = St;
        count += St;
        cout << "St : " << St << endl;
        // cout << "S " << St << endl;
        // cout << i << endl;
    }
    cout << "paths calculated" << endl;
    cout << "mean paths : " << count/N_PATHS << endl;



    // simulateOptionPrice<<<1, N_PATHS>>>( d_optionPriceGPU,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt);
    // cudaDeviceSynchronize();


    float *a;
    a = (float *)malloc(N_PATHS * sizeof(float));
    float *d_a;
    testCUDA(cudaMalloc((void **)&d_a,N_PATHS*sizeof(float)));
    cudaMemcpy(d_a, a, N_PATHS * sizeof(int), cudaMemcpyHostToDevice);

    simulateOptionPrice<<<1, N_PATHS>>>( d_a,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt);
    cudaDeviceSynchronize();

    cudaMemcpy(a, d_a, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i<N_PATHS; i++){
        cout << "GPU St : " << a[i] << endl;
    }


    // float *d_optionPriceGU;
    // testCUDA(cudaMalloc(&d_optionPriceGPU,N_PATHS*sizeof(float)));

    // int blockSize = 256; // You can adjust this based on your GPU's capability
    // int numBlocks = (N_PATHS + blockSize - 1) / blockSize;

    // cout << "nb block" << numBlocks << endl;

    // initializeArray<<<numBlocks, blockSize>>>(d_optionPriceGPU, N_PATHS, 6.0f);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    // }
    // testCUDA(cudaDeviceSynchronize());
    // float *h_optionPriceGPU = new float[N_PATHS];
    // testCUDA(cudaMemcpy(h_optionPriceGPU, d_optionPriceGPU,N_PATHS*sizeof(float),cudaMemcpyDeviceToHost));

    // for(int i = 0; i<N_PATHS; i++){
    //     cout << "GPU St : " << h_optionPriceGPU[i] << endl;
    // }
    // cout << "mean paths GPU : " << mean_priceGPU/N_PATHS << endl;


    testCUDA(cudaFree(d_randomData));
    curandDestroyGenerator(gen);

	return 0;
}


