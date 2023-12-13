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
    // option_data.N_PATHS = 100000;
    option_data.N_PATHS = 100000;
    option_data.N_PATHS_INNER = 1000;
    option_data.N_STEPS = 100;
    option_data.step = option_data.T / static_cast<float>(option_data.N_STEPS);

    int threadsPerBlock = 1024;

    // Copy option data to constant memory
    cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    // printOptionData(option_data);

    // getDeviceProperty();


    wrapper_cpu_option_vanilla(option_data, threadsPerBlock);
    wrapper_cpu_bullet_option(option_data, threadsPerBlock);

    wrapper_gpu_option_vanilla(option_data, threadsPerBlock);
    wrapper_gpu_bullet_option(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);

    // int number_blocks = get_max_blocks(threadsPerBlock);
    // printf("Computing nmc option price with %d blocks.\n", number_blocks);
    // wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, number_blocks);


    /* -------------------------------------------------------------------------- */
    /*                                 Q1 Testing                                 */
    /* -------------------------------------------------------------------------- */
    // Compute the difference between our vanilla option price and block scholes
    // depending on the number of paths used
    float call_price = 0.0;
    black_scholes_CPU(
        call_price,
        option_data.S0,
        option_data.K,
        option_data.T,
        option_data.r,
        option_data.v
    );

    float start = 1;
    float end = 1000;
    // float end = 1000001;
    int n = 1001;
    std::vector<float> N_TRAJ = linspace(start, end, n);
    std::vector<float> gpu_prices(n);


    // printf("=========================================================\n");
    // printf("=                       Question 1                      =\n");
    // printf("=========================================================\n");
    // printf("= Computing difference between Black-Scholes and Monte Carlo");
    // printf("= Using trajectory values in linspace(%.0f, %.0f, %d)\n", start, end, n);
    // printf("=========================================================\n");

    // printf("n_traj,real_value,estimated_value\n");
    // for (int i = 0; i < n; i++) {
    //     option_data.N_PATHS = N_TRAJ[i];
    //     cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    //     gpu_prices[i] = wrapper_gpu_option_vanilla(option_data, threadsPerBlock, true);
    //     printf("%.0f,%f,%f\n", N_TRAJ[i], call_price, gpu_prices[i]);
    // }


    /* -------------------------------------------------------------------------- */
    /*               Question 2: Compare performance of CPU and GPU               */
    /* -------------------------------------------------------------------------- */
    printf("=========================================================\n");
    printf("=                       Question 2                      =\n");
    printf("=========================================================\n");
    printf("= Comparing performance between CPU and GPU\n");
    printf("=========================================================\n");

    // printf("n_traj,real_value,estimated_value\n");
    // for (int i = 0; i < n; i++) {
    //     option_data.N_PATHS = N_TRAJ[i];
    //     cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    //     gpu_prices[i] = wrapper_gpu_option_vanilla(option_data, threadsPerBlock, true);
    //     printf("%.0f,%f,%f\n", N_TRAJ[i], call_price, gpu_prices[i]);
    // }











    /* -------------------------------------------------------------------------- */
    /*                                 Q3 Testing                                 */
    /* -------------------------------------------------------------------------- */
    // vector<float> block_range = linspace(400, 2000, 8);

    // Clock clock;
    // // wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, number_blocks);
    // printf("=========================================================\n");
    // printf("=                       Timing                          =\n");
    // printf("=========================================================\n");
    // printf("= Running %d[%d] paths with %d steps \n", option_data.N_PATHS, option_data.N_PATHS_INNER, option_data.N_STEPS);
    // printf("= Max blocks: %d\n", number_blocks);
    // printf("= Testing across linspace(%.0f, %.0f, %d)\n", block_range[0], block_range.back(), block_range.size());
    // printf("=========================================================\n");


    // printf("i,n_blocks,time_optimal,time_one_point,time_one_kernel\n");
    // // for (int i = 1; i < 11; i++)    {
    // int i = 0;

    // for (float nb : block_range) {

    //     // float nb = i * number_blocks/10;

    //     // wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, nb);

    //     // Timing block
    //     auto optimal_callback = [&] () {
    //         wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, nb, true);
    //     };

    //     auto one_point_callback = [&] () {
    //         wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, nb, true);
    //     };

    //     auto one_kernel_callback = [&] () {
    //         wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, nb, true);
    //     };

    //     float time_optimal = clock.time_fn(optimal_callback, 5);
    //     float time_one_point = clock.time_fn(one_point_callback, 5);
    //     float time_one_kernel = clock.time_fn(one_kernel_callback, 5);

    //     printf("%d,%d,%f,%f,%f\n", i, (int) nb, time_optimal, time_one_point, time_one_kernel);
    //     i ++;

    // }

    // float callResult = 0.0f;
    // black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    // std::cout << endl << "call Black Scholes : " << callResult << endl;

    return 0;
}