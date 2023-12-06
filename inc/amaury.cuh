#include <iostream>
#include <cuda_runtime.h>
#include  "pricinghost.hpp"
#include <random>
#include <curand.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <math.h>
#include "BlackandScholes.hpp"

using namespace std;
namespace cg = cooperative_groups;

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

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

__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n) {

    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float sdata[1024];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);

    }

    // for (unsigned int s=1; s < blockDim.x; s*= 2) {
    //     if (tid % (2 * s) == 0) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    // }

  // write result for this block to global mem
  if (tid == 0){

    g_odata[blockIdx.x] = sdata[0];

  }
}



__global__ void reduce4(float *g_idata, float *g_odata, unsigned int n) {
  // Handle to thread block group
  const int blockSize = 1024;
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float sdata[1024];


  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float mySum = (i < n) ? g_idata[i] : 0;

  if (i + 1024 < n) mySum += g_idata[i + 1024];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

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

__global__ void reduce5(float *g_idata, float *g_odata, unsigned int n) {
  // Handle to thread block group
  const int blockSize = 1024;
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

  float mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ( (blockSize >= 1024) && (tid < 512)) {
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




__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n, bool nIsPow2) {
  // Handle to thread block group
  const int blockSize = 1024;
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float sdata[blockSize];
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

// do reduction in shared mem
  if ( (blockSize >= 1024) && (tid < 512)) {
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








__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128)
__global__ void BlackScholesGPU(
    float2 * __restrict d_CallResult,
    float2 * __restrict d_PutResult,
    float2 * __restrict d_StockPrice,
    float2 * __restrict d_OptionStrike,
    float2 * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

     // Calculating 2 options per thread to increase ILP (instruction level parallelism)
    if (opt < (optN / 2))
    {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(
            callResult1,
            putResult1,
            d_StockPrice[opt].x,
            d_OptionStrike[opt].x,
            d_OptionYears[opt].x,
            Riskfree,
            Volatility
        );
        BlackScholesBodyGPU(
            callResult2,
            putResult2,
            d_StockPrice[opt].y,
            d_OptionStrike[opt].y,
            d_OptionYears[opt].y,
            Riskfree,
            Volatility
        );
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
	 }
}




void generateRandomArray(float *d_randomData, float *h_randomData, int N_PATHS, int N_STEPS, unsigned long long seed = 1234ULL){

    // create generator all fill array with generated values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_randomData, N_PATHS * N_STEPS, 0.0, 1.0);
    testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
    curandDestroyGenerator(gen);

}

void generate_random_array(float *d_randomData, float *h_randomData, int length, unsigned long long seed = 1234ULL) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, d_randomData, length, 0.0, 1.0);
    testCUDA(cudaMemcpy(h_randomData, d_randomData, length * sizeof(float), cudaMemcpyDeviceToHost));
    curandDestroyGenerator(gen);
}




__global__ void simulateOptionPriceGPU(float *d_optionPriceGPU, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_PATHS) {
        float St = S0;
        float G;
        for(int i = 0; i < N_STEPS; i++){
            G = d_randomData[idx*i];
            // cout << "G : " << G << endl;
            St *= exp((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
        }

        // // Calculate the payoff
        d_optionPriceGPU[idx] = max(St - K, 0.0f);


    }
}


__global__ void simulateOptionPriceOneBlockGPUSumReduce(float *d_optionPriceGPU, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt, float *output) {
    int stride = blockDim.x;
    int idx = threadIdx.x;
    int tid = threadIdx.x;

    // Shared memory for the block
    __shared__ float sdata[1024];
    float sum = 0.0f;

    if(idx < N_PATHS) {
        sdata[tid] = 0.0f;

        while(idx < N_PATHS){
            float St = S0;
            float G;
            for(int i = 0; i < N_STEPS; i++){
                G = d_randomData[idx*N_STEPS + i];
                // cout << "G : " << G << endl;
                St *= exp((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
            }
            sum += max(St - K,0.0f);
            idx += stride;
        }
    // Load input into shared memory
        sdata[tid] = (tid < N_PATHS) ? sum : 0;

        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = 1024 / 2; s > 0; s >>= 1) {
            if (tid < s && (tid + s) < N_PATHS) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to output
        if (tid == 0){
            output[0] = sdata[0];
            }
        }
}


__global__ void simulateOptionPriceMultipleBlockGPU(float *d_simulated_payoff, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N_PATHS) {
            float St = S0;
            float G;
            for(int i = 0; i < N_STEPS; i++){
                G = d_randomData[idx*N_STEPS + i];
                St *= expf((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
            }
            d_simulated_payoff[idx] = max(St - K,0.0f);
        }
    }

__global__ void simulateBulletOptionPriceMultipleBlockGPU(float *d_simulated_payoff, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt, float B, float P1, float P2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < N_PATHS) {
    int count = 0;
    float St = S0;
    float G;
    for(int i = 0; i < N_STEPS; i++){
        G = d_randomData[idx*N_STEPS + i];
        St *= expf((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
        if(B > St) count +=1;
        }
    if((count >= P1) && (count <= P2)){
      d_simulated_payoff[idx] = max(St - K,0.0f);
    } else {
      d_simulated_payoff[idx] = 0.0f;
    }
  }
}


void getDeviceProperty(){

    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    printf("The number of devices available is %i GPUs \n", count);
    cudaGetDeviceProperties(&prop, count-1);
    printf("Name: %s\n", prop.name);
    printf("Global memory size in bytes: %ld\n", prop.totalGlobalMem);
    printf("Shared memory size per block: %ld\n", prop.sharedMemPerBlock);
    printf("Number of registers per block: %d\n", prop.regsPerBlock);
    printf("Number of threads in a warp: %d\n", prop.warpSize);
    printf("Maximum number of threads that can be launched per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads that can be launched: %d x %d x %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum grid size: %d x %d x %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total constant memory: %ld\n", prop.totalConstMem);
    printf("Major compute capability: %d\n", prop.major);
    printf("Minor compute capability: %d\n", prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Maximum 1D texture memory: %d\n", prop.maxTexture1D);
    printf("Could we overlap? %d\n", prop.deviceOverlap);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Is there a limit for kernel execution? %d\n", prop.kernelExecTimeoutEnabled);
    printf("Is my GPU a chipsest? %d\n", prop.integrated);
    printf("Can we map the host memory? %d\n", prop.canMapHostMemory);
    printf("Can we launch concurrent kernels? %d\n", prop.concurrentKernels);
    printf("Do we have ECC memory? %d\n", prop.ECCEnabled);

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


/**
 * @brief Allocate space for and fill the elements of an array with size `length`.
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

// // Let's create a structure that wraps up the parameters of our simulation
// struct


/**
 * @brief Test different reduction methods
 *
 * @param n_trajectories
 * @param n_blocks
 * @param reduction
 */
void testReductions(int n_trajectories, int n_steps, int n_blocks = 1, int reduction = 0) {

    // Parameters:

    // Defaults
    const float risk_free_rate = 0.1;      // r
    const float initial_spot_price = 100;  // x_0
    const float contract_strike = 100;     // K
    const float contract_maturity = 1;     // T
    const float volatitility = 0.2;        // sigma

    // Uninitalized
    float barrier = 0;

    size_t seed = 1234ULL;

    // Variables needed for simulations


    // Step one: Get random data from the gpu
    float *d_randomData, *h_randomData;

    initRandomArray(&d_randomData, &h_randomData, n_trajectories * n_steps, seed);

    // Step two: Simulate different trajectories
    // - need to simulate trajectories on the gpu and the cpu


    // Intended effect: Allocate


    // I want to call different reductions, both on the CPU and GPU

    // First step: wrap up the CPU code into a function
}


// Compute a trajectory using the CPU
// For now we can accept the random data


// We should wrap this up in a new executable
