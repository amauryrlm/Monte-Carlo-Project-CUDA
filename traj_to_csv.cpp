#include <iostream>
#include <fstream>
#include <format>
#include <string>

#include "monte_carlo.hpp"

using namespace monte_carlo;


int main(int argc, char* argv[]) {

    int n_traj = 10;

    if (argc > 1) {
        n_traj = std::stoi(argv[1]);
    }

    // Let's create multiple different trajectories, of different lengths

    std::vector<int> N {1, 2, 3, 5, 10, 25, 50, 100, 1000};

    std::cout << std::format("Spawning {} trajectories with the following lengths:\n", n_traj);
    print_vector(N);

    for (int n_steps : N) {


        std::ofstream myfile;
        myfile.open(std::format("out_{:04}.csv", n_steps));

        // Simulate n_trajectories of length n_steps
        auto trajectories = replicate<std::vector<double>>(
                                [&] { return simulate_trajectory(100.0, n_steps); },
                                n_traj
                            );

        // Create the header
        myfile << "t,";

        for (int i = 0; i < trajectories.size() - 1; i++) {
            myfile << std::format("traj_{},", i);
        }
        myfile << std::format("traj_{}\n", trajectories.size() - 1);

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