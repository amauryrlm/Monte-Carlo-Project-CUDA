#include "trajectories.cuh"
#include "BlackandScholes.hpp"
#include "reduce.cuh"
#include "tool.cuh"
#include "option_price.hpp"
#include "nmc.cuh"

float wrapper_cpu_option_vanilla(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;

    float *d_randomData, *h_randomData;
    testCUDA(cudaMalloc(&d_randomData, N_PATHS * N_STEPS * sizeof(float)));
    h_randomData = (float *) malloc(N_PATHS * N_STEPS * sizeof(float));
    generateRandomArray(d_randomData, h_randomData, N_PATHS, N_STEPS);


    float optionPriceCPU = 0.0f;
    simulateOptionPriceCPU(&optionPriceCPU, h_randomData, option_data);

    cout << endl;
    cout << "Average CPU : " << optionPriceCPU << endl << endl;
    free(h_randomData);
    cudaFree(d_randomData);

    return optionPriceCPU;
}

float wrapper_gpu_option_vanilla(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    // generate states array for random number generation
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, blocksPerGrid * sizeof(float)));
    float *h_odata = (float *) malloc(blocksPerGrid * sizeof(float));

    simulateOptionPriceMultipleBlockGPUwithReduce<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_odata[i];
    }
    float optionPriceGPU = expf(-option_data.r * option_data.T) * sum / N_PATHS;
    cout << "Average GPU : " << optionPriceGPU << endl << endl;
    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;
}

float wrapper_gpu_bullet_option(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, blocksPerGrid * sizeof(float)));
    float *h_odata = (float *) malloc(blocksPerGrid * sizeof(float));
    CHECK_MALLOC(h_odata);

    simulateBulletOptionPriceMultipleBlockGPU<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_odata[i];
    }
    float optionPriceGPU = expf(-option_data.r * option_data.T) * sum / static_cast<float>(N_PATHS);
    cout << "Average GPU bullet option : " << optionPriceGPU << endl << endl;

    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;

}

float wrapper_gpu_bullet_option_atomic(OptionData option_data, int threadsPerBlock) {

    int N_PATHS = option_data.N_PATHS;
    int blocksPerGrid = (option_data.N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    curandState *d_states;
    testCUDA(cudaMalloc(&d_states, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234);

    float *d_odata;
    testCUDA(cudaMalloc(&d_odata, sizeof(float)));
    float *h_odata = (float *) malloc(sizeof(float));
    CHECK_MALLOC(h_odata);

    simulateBulletOptionPriceMultipleBlockGPUatomic<<<blocksPerGrid, threadsPerBlock>>>(d_odata, d_states);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaDeviceSynchronize();
    testCUDA(cudaMemcpy(h_odata, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

    float optionPriceGPU = expf(-option_data.r * option_data.T) * h_odata[0] / static_cast<float>(N_PATHS);
    cout << "Average GPU bullet option atomic : " << optionPriceGPU << endl << endl;

    free(h_odata);
    cudaFree(d_odata);
    cudaFree(d_states);
    return optionPriceGPU;

}


float
wrapper_gpu_bullet_option_nmc_one_point_one_block(OptionData option_data, int threadsPerBlock, int number_of_blocks) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states_outter, *d_states_inner;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);


    testCUDA(cudaMalloc(&d_states_outter, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states_outter, 1234);


    simulate_outer_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_option_prices, d_states_outter, d_stock_prices,
                                                                    d_sums_i);

    cudaDeviceSynchronize();
    cudaFree(d_states_outter);


    testCUDA(cudaMalloc(&d_states_inner, number_of_blocks * threadsPerBlock * sizeof(curandState)));

    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states_inner, 1235);


    size_t freeMem2;
    size_t totalMem2;
    testCUDA(cudaMemGetInfo(&freeMem2, &totalMem2));


    std::cout << "Free memory : " << freeMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Total memory : " << totalMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Used memory : " << (totalMem2 - freeMem2) / 1024 / 1024 << " MB\n";


    compute_nmc_one_block_per_point<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states_inner,
                                                                           d_stock_prices, d_sums_i);


    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, number_of_options * sizeof(int), cudaMemcpyDeviceToHost));

    //compute average of option prices
    float sum = 0.0f;
    for (int i = 0; i < number_of_options; i++) {
        sum += h_option_prices[i];
    }
    float callResult = sum / static_cast<float>(number_of_options);
    cout << "Average GPU bullet option nmc one point per block : " << callResult
         << endl << endl;


    free(h_option_prices);
    free(h_stock_prices);
    free(h_sums_i);
    cudaFree(d_option_prices);
    cudaFree(d_stock_prices);
    cudaFree(d_sums_i);
    cudaFree(d_states_inner);
    cudaFree(d_states_outter);


    return callResult;

}


float wrapper_gpu_bullet_option_nmc_one_kernel(OptionData option_data, int threadsPerBlock, int number_of_blocks) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);

    testCUDA(cudaMalloc(&d_states, number_of_blocks * threadsPerBlock * sizeof(curandState)));
    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states, 1234);

    compute_nmc_one_block_per_point_with_outter<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states,
                                                                                       d_stock_prices, d_sums_i);
    cudaDeviceSynchronize();


    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, number_of_options * sizeof(int), cudaMemcpyDeviceToHost));


    float sum = 0.0f;
    for (int i = 0; i < N_PATHS * N_STEPS; i++) {
        sum += h_option_prices[i];
    }


    float callResult = sum / static_cast<float>(number_of_options);
    cout << "Average GPU bullet option nmc one kernel : " << callResult
         << endl << endl;


    free(h_option_prices);
    free(h_stock_prices);
    free(h_sums_i);
    cudaFree(d_option_prices);
    cudaFree(d_stock_prices);
    cudaFree(d_sums_i);
    cudaFree(d_states);

    return 0.0f;

}

float
wrapper_gpu_bullet_option_nmc_optimal(OptionData option_data, int threadsPerBlock, int number_of_blocks) {

    int N_PATHS = option_data.N_PATHS;
    int N_STEPS = option_data.N_STEPS;
    int blocksPerGrid = (N_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    int number_of_options = N_PATHS * N_STEPS + 1;

    curandState *d_states_outter, *d_states_inner;
    float *d_option_prices, *d_stock_prices;
    int *d_sums_i;
    testCUDA(cudaMalloc(&d_option_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_stock_prices, number_of_options * sizeof(float)));
    testCUDA(cudaMalloc(&d_sums_i, number_of_options * sizeof(int)));
    float *h_option_prices = (float *) malloc(number_of_options * sizeof(float));
    float *h_stock_prices = (float *) malloc(number_of_options * sizeof(float));
    int *h_sums_i = (int *) malloc(number_of_options * sizeof(int));
    CHECK_MALLOC(h_option_prices);
    CHECK_MALLOC(h_stock_prices);
    CHECK_MALLOC(h_sums_i);


    testCUDA(cudaMalloc(&d_states_outter, N_PATHS * sizeof(curandState)));
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states_outter, 1234);


    simulate_outer_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_option_prices, d_states_outter, d_stock_prices,
                                                                    d_sums_i);

    cudaDeviceSynchronize();
    cudaFree(d_states_outter);


    testCUDA(cudaMalloc(&d_states_inner, number_of_blocks * threadsPerBlock * sizeof(curandState)));

    setup_kernel<<<number_of_blocks, threadsPerBlock>>>(d_states_inner, 1235);


    size_t freeMem2;
    size_t totalMem2;
    testCUDA(cudaMemGetInfo(&freeMem2, &totalMem2));


    std::cout << "Free memory : " << freeMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Total memory : " << totalMem2 / 1024 / 1024 << " MB\n";
    std::cout << "Used memory : " << (totalMem2 - freeMem2) / 1024 / 1024 << " MB\n";


    compute_nmc_optimal<<<number_of_blocks, threadsPerBlock>>>(d_option_prices, d_states_inner,
                                                               d_stock_prices, d_sums_i);


    testCUDA(cudaMemcpy(h_option_prices, d_option_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_stock_prices, d_stock_prices, number_of_options * sizeof(float), cudaMemcpyDeviceToHost));
    testCUDA(cudaMemcpy(h_sums_i, d_sums_i, number_of_options * sizeof(int), cudaMemcpyDeviceToHost));

    //compute average of option prices
    float sum = 0.0f;
    for (int i = 0; i < number_of_options; i++) {
        sum += h_option_prices[i] / static_cast<float>(option_data.N_PATHS_INNER);
    }
    float callResult = sum / static_cast<float>(number_of_options);
    cout << "Average GPU bullet option nmc optimal : " << callResult
         << endl << endl;

    cout << "option price 0 optimal << " << h_option_prices[number_of_options - 1] * expf(-option_data.r*option_data.T) / static_cast<float>(option_data.N_PATHS) << endl;


    free(h_option_prices);
    free(h_stock_prices);
    free(h_sums_i);
    cudaFree(d_option_prices);
    cudaFree(d_stock_prices);
    cudaFree(d_sums_i);
    cudaFree(d_states_inner);
    cudaFree(d_states_outter);


    return callResult;

}


int main(void) {

    OptionData option_data;
    option_data.S0 = 100.0f;
    option_data.T = 1.0f;
    option_data.K = 100.0f;
    option_data.r = 0.1f;
    option_data.v = 0.2f;
    option_data.B = 120.0f;
    option_data.P1 = 10;
    option_data.P2 = 50;
    option_data.N_PATHS = 100000;
    option_data.N_PATHS_INNER = 1000;
    option_data.N_STEPS = 100;
    option_data.step = option_data.T / static_cast<float>(option_data.N_STEPS);

    int threadsPerBlock = 1024;

    // Copy option data to constant memory
    cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    printOptionData(option_data);

    getDeviceProperty();

    wrapper_cpu_option_vanilla(option_data, threadsPerBlock);

    wrapper_gpu_option_vanilla(option_data, threadsPerBlock);
    wrapper_gpu_bullet_option(option_data, threadsPerBlock);
    wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);

    int number_blocks = get_max_blocks(threadsPerBlock);
    printf("Computing nmc option price with %d blocks.\n", number_blocks);
    // wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, number_blocks);


    // wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, number_blocks);
    for (int i = 1; i < 11; i++)    {

        float nb = i * number_blocks/10;
        cout << endl << endl << "Iteration : " << i << " number of blocks : " << nb << endl;
        wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, nb);
    }

    float callResult = 0.0f;
    black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    cout << endl << "call Black Scholes : " << callResult << endl;

    return 0;
}