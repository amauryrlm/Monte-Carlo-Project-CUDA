#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <curand.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <math.h>
#include "BlackandScholes.hpp"
#include "reduce.cuh"

using namespace std;


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
    cout << endl << "number generated" << endl;
    testCUDA(cudaMemcpy(h_randomData, d_randomData, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "host copied" << endl;
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

__global__ void simulateBulletOptionSavePrice(float *d_simulated_paths,float *d_simulated_count, float K, float r, float T,float sigma, int N_PATHS, float *d_randomData, int N_STEPS, float S0, float dt, float sqrdt, float B, float P1, float P2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(idx < N_PATHS) {
    int count = 0;
    float St = S0;
    float G;
    for(int i = 0; i < N_STEPS; i++){
        G = d_randomData[idx*N_STEPS + i];
        St *= expf((r - (sigma*sigma)/2)*dt + sigma * sqrdt * G);
        if(B > St) count +=1;
        if((count >= P1) && (count <= P2)){
          d_simulated_paths[idx * N_STEPS + i] = St;
        } else {
          d_simulated_paths[idx * N_STEPS + i] = 0.0f;
        }
        d_simulated_count[idx * N_STEPS + i] = count;
  }
}
}






int main(void) {



// declare variables and constants
    unsigned int N_PATHS = 1000000;
    const size_t N_STEPS = 100;
    const float T = 1.0f;
    const float K = 100.0f;
    const float B = 110.0f;
    const float S0 = 100.0f;
    const float sigma = 0.2f;
    const float r = 0.05f;
    float dt = float(T)/float(N_STEPS);
    float sqrdt = sqrt(dt);
    vector<float> s(N_PATHS);
    int threadsPerBlock = 1024;
    unsigned int maxThreads = 1024;
    int P1 = 10;
    int P2 = 50;

    getDeviceProperty();

    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;

    cout << "number of paths : " << N_PATHS << endl;
    cout << "number of steps : " << N_STEPS << endl;


    float *d_randomData, *h_randomData, *simulated_paths_cpu;
    testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));
    h_randomData = (float *)malloc(N_PATHS * N_STEPS*sizeof(float));
    simulated_paths_cpu = (float *)malloc(N_PATHS *sizeof(float));
    generateRandomArray(d_randomData, h_randomData, N_PATHS, N_STEPS);


    cout << "random  " << h_randomData[0] << endl;

    float optionPriceCPU = 0.0f;
    simulateOptionPriceCPU(&optionPriceCPU,  N_PATHS,  N_STEPS,  h_randomData,  S0,  sigma,  sqrdt,  r, K, dt,simulated_paths_cpu);

    cout << endl;
    cout << "Average CPU : " << optionPriceCPU << endl << endl;



//--------------------------------GPU WITH ONE BLOCK ----------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

    float *h_optionPriceGPU, *output;
    h_optionPriceGPU = (float *)malloc(N_PATHS * sizeof(float));
    output = (float *)malloc(sizeof(float));
    float *d_optionPriceGPU, *d_output;

    testCUDA(cudaMalloc((void **)&d_optionPriceGPU,N_PATHS*sizeof(float)));
    testCUDA(cudaMalloc((void **)&d_output,sizeof(float)));

    simulateOptionPriceOneBlockGPUSumReduce<<<1, 1024>>>( d_optionPriceGPU,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt, d_output);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaMemcpy(h_optionPriceGPU, d_optionPriceGPU, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cout << endl;

    cout << "Average GPU one block : " << output[0]/N_PATHS << endl ;

    cudaFree(d_optionPriceGPU);
    cudaFree(d_output);
    free(h_optionPriceGPU);
    free(output);


//--------------------------------GPU WITH MULTIPLE BLOCK ----------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------


    int threads = (N_PATHS < maxThreads * 2) ? nextPow2((N_PATHS + 1) / 2) : maxThreads;
    int blocks = (N_PATHS + (threads * 2 - 1)) / (threads * 2);


    float *output3, *d_optionPriceGPU3, *d_output3;
    output3 = (float *)malloc(blocks * sizeof(float));

    testCUDA(cudaMalloc((void **)&d_optionPriceGPU3,N_PATHS*sizeof(float)));
    testCUDA(cudaMalloc((void **)&d_output3,blocks * sizeof(float)));

    cout << "number of blocks : " << blocksPerGrid << endl;
    cout << "number of threads : " << threadsPerBlock << endl;

    simulateOptionPriceMultipleBlockGPU<<<blocksPerGrid,threadsPerBlock>>>( d_optionPriceGPU3,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt);
    cudaError_t error3 = cudaGetLastError();
    if (error3 != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error3));
        return -1;
    }

    cout << "number of blocks" << blocks << endl;
    cout << "number of threads" << threads << endl;

    reduce3<<<blocks,threads>>>(d_optionPriceGPU3,d_output3,N_PATHS);
    error3 = cudaGetLastError();
    if (error3 != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error3));
        return -1;
    }


    cudaDeviceSynchronize();

    testCUDA(cudaMemcpy(output3, d_output3, blocks * sizeof(float), cudaMemcpyDeviceToHost));


    cout << endl;
    float sum = 0.0f;
    for(int i=0; i<blocks; i++){
        sum+=output3[i];
    }
    cout<< "result gpu cuda computed " << sum/N_PATHS << endl;

    cudaFree(d_optionPriceGPU3);
    cudaFree(d_output3);
    free(output3);





//--------------------------------BLACK SCHOLES FORMULA ----------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

    float callResult = 0.0f;
    black_scholes_CPU(callResult,S0, K, T, r,  sigma);
    cout << "call BS : " << callResult << endl;






//-------------------------------BULLET OPTION WITH MULTIPLE BLOCKS ------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

  float *d_simulated_payoff_bullet, *h_simulated_payoff_bullet, *h_output4, *d_output4;
  testCUDA(cudaMalloc((void **)&d_simulated_payoff_bullet, N_PATHS * sizeof(float)));
  testCUDA(cudaMalloc((void **)&d_output4, blocks * sizeof(float)));
  h_output4 = (float *)malloc(blocks * sizeof(float));
  h_simulated_payoff_bullet = (float *)malloc(N_PATHS * sizeof(float));

  simulateBulletOptionPriceMultipleBlockGPU<<<blocksPerGrid,threadsPerBlock>>>( d_simulated_payoff_bullet,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt, B, P1, P2);
  cudaError_t error4 = cudaGetLastError();
  if (error4 != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error4));
      return -1;
  }
  cudaDeviceSynchronize();
  testCUDA(cudaMemcpy(h_simulated_payoff_bullet, d_simulated_payoff_bullet, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost));

  reduce3<<<blocks,threads>>>(d_simulated_payoff_bullet,d_output4,N_PATHS);
  error4 = cudaGetLastError();
  if (error4 != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error4));
      return -1;
  }
  cudaDeviceSynchronize();
  testCUDA(cudaMemcpy(h_output4, d_output4, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float sum4 = 0.0f;
  for(int i=0; i<blocks; i++){
      sum4+=h_output4[i];
  }
  cout<< "result gpu cuda computed bullet option " << sum4/N_PATHS << endl;



  //-------------------------------BULLET OPTION WITH MULTIPLE BLOCKS AND SAVE PATHS------------------------------------------------------------
  //-------------------------------------------------------------------------------------------------------------------------------------------

  float *d_simulated_paths, *d_simulated_count, *h_simulated_paths, *h_simulated_count;
  testCUDA(cudaMalloc((void **)&d_simulated_paths, N_PATHS * N_STEPS * sizeof(float)));
  testCUDA(cudaMalloc((void **)&d_simulated_count, N_PATHS * N_STEPS * sizeof(float)));
  h_simulated_paths = (float *)malloc(N_PATHS * N_STEPS * sizeof(float));
  h_simulated_count = (float *)malloc(N_PATHS * N_STEPS * sizeof(float));

  simulateBulletOptionSavePrice<<<blocksPerGrid,threadsPerBlock>>>( d_simulated_paths, d_simulated_count,  K,  r,  T, sigma,  N_PATHS,  d_randomData,  N_STEPS, S0, dt, sqrdt, B, P1, P2);
  cudaError_t error5 = cudaGetLastError();
  if (error5 != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error5));
      return -1;
  }
  cudaDeviceSynchronize();
  testCUDA(cudaMemcpy(h_simulated_paths, d_simulated_paths, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost));
  testCUDA(cudaMemcpy(h_simulated_count, d_simulated_count, N_PATHS * N_STEPS * sizeof(float), cudaMemcpyDeviceToHost)); 

  
  
  cudaFree(d_simulated_payoff_bullet);
  cudaFree(d_output4);
  free(h_output4);
  cudaFree(d_randomData);



	return 0;
}


