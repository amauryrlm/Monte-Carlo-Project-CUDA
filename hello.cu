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
    option_data.N_PATHS_INNER = 10000;
    option_data.N_STEPS = 100;
    option_data.step = option_data.T / static_cast<float>(option_data.N_STEPS);

    int threadsPerBlock = 1024;

    // Copy option data to constant memory
    cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    // printOptionData(option_data);

    getDeviceProperty();


    // wrapper_cpu_option_vanilla(option_data, threadsPerBlock);
    // wrapper_cpu_bullet_option(option_data, threadsPerBlock);

    wrapper_gpu_option_vanilla(option_data, threadsPerBlock, 5000);
    // wrapper_gpu_bullet_option(option_data, threadsPerBlock);
    // wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);

    // int number_blocks = get_max_blocks(threadsPerBlock);
    // printf("Computing nmc option price with %d blocks.\n", number_blocks);
    // wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, 5000);


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
    int n = 1000;
    std::vector<float> N_TRAJ = linspace(start, end, n);
    std::vector<float> gpu_prices(n);
    int number_blocks = 400;

    printf("=========================================================\n");
    printf("=                       Question 1                      =\n");
    printf("=========================================================\n");
    printf("= Computing difference between Black-Scholes and Monte Carlo");
    printf("= Using trajectory values in linspace(%.0f, %.0f, %d)\n", start, end, n);
    printf("=========================================================\n");

    printf("n_traj,real_value,estimated_value\n");
    for (int i = 0; i < n; i++) {
        option_data.N_PATHS = N_TRAJ[i];
        cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
        gpu_prices[i] = wrapper_gpu_option_vanilla(option_data, threadsPerBlock, number_blocks);
        printf("%.0f,%f,%f\n", N_TRAJ[i], call_price, gpu_prices[i]);
    }


    /* -------------------------------------------------------------------------- */
    /*               Question 2: Compare performance of CPU and GPU               */
    /* -------------------------------------------------------------------------- */
    // printf("=========================================================\n");
    // printf("=                       Question 2                      =\n");
    // printf("=========================================================\n");
    // printf("= Comparing performance between CPU and GPU\n");
    // printf("=========================================================\n");

    // // Iterating through number of trajectories, compare the speeds between CPU and gpu
    // Clock clock;
    // int n_q2 = 100;
    // float n_traj_0 = 1000;
    // float n_traj_F = 100000;
    // threadsPerBlock = 1024;

    // vector<float> TRAJ = linspace(n_traj_0, n_traj_F, n_q2);

    // // Compare pricing functions
    // printf("n_traj,time_cpu_vanilla,time_gpu_vanilla,time_cpu_bullet,time_gpu_bullet\n");
    // for (int i = 0; i < n_q2; i++) {

    //     option_data.N_PATHS = TRAJ[i];
    //     cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));

    //     auto cpu_bullet_fn = [&] () {
    //         wrapper_cpu_bullet_option(option_data, threadsPerBlock);
    //     };

    //     auto gpu_bullet_fn = [&] () {
    //         wrapper_gpu_bullet_option_atomic(option_data, threadsPerBlock);
    //     };

    //     auto cpu_vanilla_fn = [&] () {
    //         wrapper_cpu_option_vanilla(option_data, threadsPerBlock);
    //     };

    //     auto gpu_vanilla_fn = [&] () {
    //         wrapper_gpu_option_vanilla(option_data, threadsPerBlock);
    //     };

    //     float cpu_bullet = clock.time_fn(cpu_bullet_fn);
    //     float gpu_bullet = clock.time_fn(gpu_bullet_fn);
    //     float cpu_vanilla = clock.time_fn(cpu_vanilla_fn);
    //     float gpu_vanilla = clock.time_fn(gpu_vanilla_fn);

    //     printf("%d,%f,%f,%f,%f\n", (int) TRAJ[i], cpu_vanilla, gpu_vanilla, cpu_bullet, gpu_bullet);
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



    /* -------------------------------------------------------------------------- */
    /*                                 Q3 Testing                                 */
    /* -------------------------------------------------------------------------- */
    // vector<float> n_traj = linspace(50000, 150000, 9);

    // Clock clock;
    // // wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, number_blocks);
    // printf("=========================================================\n");
    // printf("=                       Timing                          =\n");
    // printf("=========================================================\n");
    // printf("= Running %d[%d] paths with %d steps \n", option_data.N_PATHS, option_data.N_PATHS_INNER, option_data.N_STEPS);
    // printf("= Using 400 Blocks for each callback\n");
    // printf("= Max blocks: %d\n", get_max_blocks(threadsPerBlock));
    // printf("= Testing across linspace(%.0f, %.0f, %d)\n", n_traj[0], n_traj.back(), n_traj.size());
    // printf("=========================================================\n");


    // printf("i,n_blocks,n_traj,time_optimal,time_one_point,time_one_kernel\n");
    // // for (int i = 1; i < 11; i++)    {
    // int i = 0;

    // for (float t : n_traj) {

    //     // float nb = i * number_blocks/10;

    //     // wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, nb);

    //     // Timing block
    //     auto optimal_callback = [&] () {
    //         option_data.N_PATHS = t;
    //         cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    //         wrapper_gpu_bullet_option_nmc_optimal(option_data, threadsPerBlock, 400);
    //     };

    //     auto one_point_callback = [&] () {
    //         option_data.N_PATHS = t;
    //         cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    //         wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, threadsPerBlock, 400);
    //     };

    //     auto one_kernel_callback = [&] () {
    //         option_data.N_PATHS = t;
    //         cudaMemcpyToSymbol(d_OptionData, &option_data, sizeof(OptionData));
    //         wrapper_gpu_bullet_option_nmc_one_kernel(option_data, threadsPerBlock, 400);
    //     };

    //     float time_optimal = clock.time_fn(optimal_callback, 1);
    //     float time_one_point = clock.time_fn(one_point_callback, 1);
    //     float time_one_kernel = clock.time_fn(one_kernel_callback, 1);

    //     printf("%d,%d,%d,%f,%f,%f\n", i, (int) 400, (int) t, time_optimal, time_one_point, time_one_kernel);
    //     i ++;

    // }




    // float callResult = 0.0f;
    // black_scholes_CPU(callResult, option_data.S0, option_data.K, option_data.T, option_data.r, option_data.v);
    // std::cout << endl << "call Black Scholes : " << callResult << endl;

    return 0;
}