//

#include "testing.cuh"

#include <iostream>
#include <stdio.h>


__global__ void print_gpu_random(float * d_random_array) {
    for (int i = 0; i < 10; i++) {
        printf("%f\n", d_random_array[i]);
    }

    __syncthreads();
}

int main() {
    std::cout << "Hello from testing suite!\n";


    // Initialize a new set of parameters
    SimulationParameters default_parameters;

    std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";


    // Now we want to launch multiple simulations from this central object.

    // Print the first five elements of our random array
    printf("Host\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n", default_parameters.h_random_array[i]);
    }

    printf("Device\n");

    // Print from the kernel's GPU
    print_gpu_random<<<1, 1>>>(default_parameters.d_random_array);

    cudaDeviceSynchronize();


}