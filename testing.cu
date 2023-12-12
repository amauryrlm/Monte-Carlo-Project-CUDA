//

#include "testing.cuh"

#include <iostream>
#include <stdio.h>
#include <fstream>


__global__ void print_gpu_random(float * d_random_array) {
    for (int i = 0; i < 10; i++) {
        printf("%f, ", d_random_array[i]);
    }

    printf("\n");

    __syncthreads();
}

void test_outer(int n_traj, int n_steps, int n_threads_per_block, uint64_t seed = 1234, float volatility = 0.2f) {

    Simulation parameters(n_traj, n_steps, volatility);
    // std::cout << "True reduction value across the entire cpu array: \n";
    // std::cout << parameters.sum_random_array();

    // Let's go ahead and get the outer trajectories.
    auto out = parameters.simulate_outer_trajectories(n_threads_per_block, seed);
    // for (auto el : out) {
    //     std::cout << el << "\n";
    // }

    std::cout << "Total size: " << out.size() << "\n";

    /* -------------------------------------------------------------------------- */
    /*                              Writing to a csv                              */
    /* -------------------------------------------------------------------------- */
    std::ofstream myfile;
    myfile.open("testing.csv");
    myfile << "time,trajectory,value\n";


    for (int i = 0; i < n_traj * n_steps; i ++) {
        int i_traj = i / n_steps;
        if (i % n_steps == 0) myfile << 0.0 << "," << i_traj << "," << parameters.x_0 << "\n";
        myfile << (1 + i % n_steps) * parameters.dt() << "," << i_traj << "," << out[i] << "\n";
    }
    myfile.close();

}

int main() {
    std::cout << "Hello from testing suite!\n";


    // Initialize a new set of parameters
    // Simulation default_parameters(16, 1);
    Simulation default_parameters(1024, 100);

    // std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";


    // // Now we want to launch multiple simulations from this central object.

    // // Print the first five elements of our random array
    // printf("Host\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f,", default_parameters.h_random_array[i]);
    // }
    // printf("\nDevice\n");

    // // Print from the kernel's GPU
    // print_gpu_random<<<1, 1>>>(default_parameters.d_random_array);
    // cudaDeviceSynchronize();

    // Now let's go ahead and simulate using CPU and GPU
    auto simulations = default_parameters.simulate_trajectory_cpu();

    std::cout << "Final value of CPU trajectory:" << simulations[simulations.size() - 1] << "\n";

    std::cout << "Len simulations: " << simulations.size() << "\n";

    for (int i = 3; i < 7; i++) {
        std::cout << "Testing reduction: " << i << "\n";
        auto out = default_parameters.test_reduction(1, 1024, i);
        for (auto el : out) {
            std::cout << el << "\n";
        }
    }

    // std::cout << "True reduction value across the entire cpu array: \n";
    // std::cout << default_parameters.sum_random_array();

    // // Let's go ahead and get the outer trajectories.
    // auto out = default_parameters.simulate_outer_trajectories(30);
    // for (auto el : out) {
    //     std::cout << el << "\n";
    // }

    // std::cout << "Total size: " << out.size() << "\n";

    // std::cout << default_parameters.n_steps << "\n";
    // std::cout << default_parameters.n_trajectories << "\n";

    int n_traj = 1000;
    int n_steps = 200;
    int n_threads_per_block = 10;
    float volatility = 0.5;

    test_outer(n_traj, n_steps, n_threads_per_block, 555, volatility);


}