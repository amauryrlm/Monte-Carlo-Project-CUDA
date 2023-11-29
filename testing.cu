//

#include "testing.cuh"

#include <iostream>


int main() {
    std::cout << "Hello from testing suite!\n";


    // Initialize a new set of parameters
    SimulationParameters default_parameters;

    std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";


    // Now we want to launch multiple simulations from this central object.







}