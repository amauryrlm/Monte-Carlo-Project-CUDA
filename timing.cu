#include "monte_carlo.cuh"

int main() {

    // first_gpu_simple();
    // gpu_simple_vs_threads();
    // gpu_simple_vs_threads_compact();

    // cpu_baseline();
    // gpu_baseline();
    // nmc_gpu_baseline();
    nmc_gpu_one_kernel_blocks_vs_time();
    nmc_gpu_optimal_blocks_vs_time();

}