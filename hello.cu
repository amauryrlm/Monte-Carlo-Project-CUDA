#include "monte_carlo.cuh"

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
    // printOptionData(option_data);

    // getDeviceProperty();


    // wrapper_cpu_option_vanilla(option_data, threadsPerBlock);
    // wrapper_cpu_bullet_option(option_data, threadsPerBlock);

    // wrapper_gpu_option_vanilla(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);

    // int number_blocks = get_max_blocks(threadsPerBlock);
    // printf("Computing nmc option price with %d blocks.\n", number_blocks);
    wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, 5000);


    wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, 5000);




    wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, number_blocks);


    float callResult = 0.0f;
    black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    cout << endl << "call Black Scholes : " << callResult << endl;

    return 0;
}