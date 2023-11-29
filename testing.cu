//

#include "testing.cuh"

#include <iostream>
#include <stdio.h>




int main() {
    std::cout << "Hello from testing suite!\n";


    // Initialize a new set of parameters
    SimulationParameters default_parameters;

    std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";


    // Now we want to launch multiple simulations from this central object.

    // Print the first five elements of our random array
    for (int i = 0; i < 10; i++) {
        printf("%f\n", default_parameters.h_random_array[i]);
    }






}