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
        // d_optionPriceGPU[idx] = 1.0f;
    }
}



// int main(void) {

// // declare variables and constants
//     const size_t N_PATHS = 10;
//     const size_t N_STEPS = 5;
//     const size_t N_NORMALS = N_PATHS*N_STEPS;
//     const float T = 1.0f;
//     const float K = 100.0f;
//     const float B = 95.0f;
//     const float S0 = 100.0f;
//     const float sigma = 0.2f;
//     const float mu = 0.1f;
//     const float r = 0.05f;
//     float dt = float(T)/float(N_STEPS);
//     float sqrdt = sqrt(dt);

//     vector<float> s(N_PATHS);

//     float step = 1.0f / N_STEPS;
//     float G = 0.0f;
//     std::default_random_engine generator;
//     std::normal_distribution<double> distribution(0.0, 1.0);

//     cout << "step : " << step << endl;
//     G = distribution(generator);
//     // cout << "G : " << G;






//     // generate random numbers using curand

//     //allocate array filled with random values 
//     float *d_randomData;
//     testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));

//     // create generator all fill array with generated values
//     curandGenerator_t gen;
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
//     curandGenerateNormal(gen, d_randomData, N_PATHS * N_STEPS, 0.0, 1.0);

//     cout << "number generated";


//     float h_randomData[N_PATHS * N_STEPS];
//     testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));

//     cout << "host copied" << endl;
//     cout << h_randomData[0];

//     // for(int i = 0; i < N_PATHS * N_STEPS; i++) {
//     //     cout << "random  : " << h_randomData[i] << endl;
//     // }

//     float count = 0.0f;
//     for(int i=0; i<N_PATHS;i++){
//         float St = S0;
//         for(int j=0; j<N_STEPS; j++){
//             G = h_randomData[i*j];
//             // cout << "G : " << G << endl;
//             St *= exp((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
            
//         }
//         // cout << "S before assigning " << St << endl;
//         s[i] = St;
//         count += St;
//         cout << "St : " << St << endl;
//         // cout << "S " << St << endl;
//         // cout << i << endl;
//     }
//     cout << "paths calculated" << endl;
//     cout << "mean paths : " << count/N_PATHS << endl;



//     // simulateOptionPrice<<<1, N_PATHS>>>( d_optionPriceGPU,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt);
//     // cudaDeviceSynchronize();
//     float *d_optionPriceGPU;
//     testCUDA(cudaMalloc(&d_optionPriceGPU,N_PATHS*sizeof(float)));

//     int blockSize = 256; // You can adjust this based on your GPU's capability
//     int numBlocks = (N_PATHS + blockSize - 1) / blockSize;

//     cout << "nb block" << numBlocks << endl;

//     initializeArray<<<numBlocks, blockSize>>>(d_optionPriceGPU, N_PATHS, 6.0f);
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
//     }
//     testCUDA(cudaDeviceSynchronize());
//     float *h_optionPriceGPU = new float[N_PATHS];
//     testCUDA(cudaMemcpy(h_optionPriceGPU, d_optionPriceGPU,N_PATHS*sizeof(float),cudaMemcpyDeviceToHost));

//     for(int i = 0; i<N_PATHS; i++){
//         cout << "GPU St : " << h_optionPriceGPU[i] << endl;
//     }
//     // cout << "mean paths GPU : " << mean_priceGPU/N_PATHS << endl;


//     testCUDA(cudaFree(d_randomData));
//     curandDestroyGenerator(gen);

// 	return 0;
// }

#include <stdio.h>

#define NB 16384
#define NTPB 1024

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}


// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__

__device__ void Test(int *a) {

	for (int i = 0; i < 1000; i++) {
		*a = *a + 1;
	}
}

__device__ int aGlob[NB*NTPB];					// Global variable solution

__global__ void MemComp(int *a){

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	aGlob[idx] = a[idx];

	Test(aGlob + idx);

	a[idx] = aGlob[idx];
}


int main (void){

	
	int *a, *aGPU;
	float Tim;										// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
		
	a = (int*)malloc(NB*NTPB*sizeof(int));
	testCUDA(cudaMalloc(&aGPU, NB*NTPB * sizeof(int)));

	for(int i=0; i<NB; i++){
		for(int j=0; j<NTPB; j++){
			a[j+i*NTPB] = j+i*NTPB;
		}
	}

	testCUDA(cudaMemcpy(aGPU, a, NB*NTPB*sizeof(int), cudaMemcpyHostToDevice));

	testCUDA(cudaEventRecord(start, 0));			// GPU timer instructions
	
	for(int i = 0; i<100; i++) {
		MemComp<<<NB,NTPB>>>(aGPU);
	}
	
	testCUDA(cudaEventRecord(stop, 0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&Tim,				// GPU timer instructions
		start, stop));								// GPU timer instructions

	printf("Time per execution: %f ms\n", Tim/100);
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	testCUDA(cudaMemcpy(a, aGPU, NB*NTPB*sizeof(int), cudaMemcpyDeviceToHost));
	testCUDA(cudaFree(aGPU));

	for(int i= 0; i<4; i++){
		printf("%i = %i \n", 100000 + i, a[i]);
	}

	free(a);

	return 0;
}
