//

#include "monte_carlo.hpp"
#include "amaury.cuh"

#include <iostream>


int main() {
    std::cout << "Hello from testing suite!\nk";


    // Initialize a new set of parameters
    SimulationParameters default_parameters;

    std::cout << "Default parameter has volatility: " << default_parameters.volatility() << "\n";

}