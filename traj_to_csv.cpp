#include <iostream>
#include <fstream>
// #include <format>
#include <string>
#include <cstdarg>

#include "monte_carlo.hpp"

using namespace monte_carlo;

// Create a string from a printf formatted string
std::string stringf(const char * format...) {
    char buffer [1000];

    va_list args;
    va_start(args, format);

    std::ofstream myfile;
    int n = std::sprintf(buffer, format, args);
    std::string s {buffer};
    return s;
}


int main(int argc, char* argv[]) {

    int n_traj = 10;

    if (argc > 1) {
        n_traj = std::stoi(argv[1]);
    }

    // Let's create multiple different trajectories, of different lengths

    std::vector<int> N {1, 2, 3, 5, 10, 25, 50, 100, 1000};

    std::cout << "Spawning " << n_traj << "trajectories with the following lengths:\n";
    print_vector(N);

    for (int n_steps : N) {

        char buffer[100];

        std::ofstream myfile;
        int n = std::sprintf(buffer, "out_%04d.csv", n_steps);
        std::string s {buffer};
        myfile.open(s);

        // Simulate n_trajectories of length n_steps
        auto trajectories = replicate<std::vector<double>>(
                                [&] { return simulate_trajectory(100.0, n_steps); },
                                n_traj
                            );

        // Create the header
        myfile << "t,";

        for (int i = 0; i < trajectories.size() - 1; i++) {
            myfile << stringf("traj_%d", i);
        }
        myfile << stringf("traj_%d\n", trajectories.size() - 1);

        // Populate the csv with the trajectory values
        for (int i = 0; i < n_steps + 1; i++) {

            myfile << i * (1.0 / n_steps) << ",";
            for (int j = 0; j < trajectories.size() - 1; j++) {
            // Construct a single line
                myfile << trajectories[j][i] << ",";
            }
            myfile << trajectories[trajectories.size() - 1][i] << "\n";

        }
    }

    return 0;
}