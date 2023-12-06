//

#include "testing.cuh"

#include <iostream>
#include <stdio.h>


__global__ void print_gpu_random(float * d_random_array) {
    for (int i = 0; i < 10; i++) {
        printf("%f, ", d_random_array[i]);
    }

    printf("\n");

    __syncthreads();
}

int main() {
    std::cout << "Hello from testing suite!\n";


    // Initialize a new set of parameters
    Simulation default_parameters(16, 1);

    std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";


    // Now we want to launch multiple simulations from this central object.

    // Print the first five elements of our random array
    printf("Host\n");
    for (int i = 0; i < 10; i++) {
        printf("%f,", default_parameters.h_random_array[i]);
    }
    printf("\nDevice\n");

    // Print from the kernel's GPU
    print_gpu_random<<<1, 1>>>(default_parameters.d_random_array);
    cudaDeviceSynchronize();

    // Now let's go ahead and simulate using CPU and GPU
    auto simulations = default_parameters.simulate_trajectory_cpu();

    std::cout << "Len simulations: " << simulations.size() << "\n";

    for (int i = 3; i < 7; i++) {
        std::cout << "Testing reduction: " << i << "\n";
        auto out = default_parameters.test_reduction(1, 20, i);
        for (auto el : out) {
            std::cout << el << "\n";
        }
    }

    std::cout << "True reduction value across the entire cpu array: \n";
    std::cout << default_parameters.sum_random_array();



}