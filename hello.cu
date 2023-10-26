#include <iostream>
// #include <format>
// #include <functional>
#include <cuda_runtime.h>

#include "trajectories.hpp"
#include "common.hpp"
#include "Xoshiro.hpp"


using namespace monte_carlo;

// template<class Sumarize>

__global__ void myKernel(void) {
}
int main(void) {

	myKernel <<<1, 1>>>();
    // Try and get a trajectory
    // auto traj = simulate_trajectory(x0, n);
    // std::cout << "Simulated a trajectory!\n";
    // float endpoint = compute_trajectory_endpoint(x0, n);

    int n_traj = 1000;
    // std::cout << "Printing a single trajectory!\n";
    // print_vector(traj);

    auto fn = [&] (int n) {

        float x0 = 100.0;

        auto traj_endpoints = compute_n_trajectory_endpoints(x0, n_traj, n);
        printf("%d\t%.2f\t%.2f\t%.2f\n", n, mean(traj_endpoints), std::sqrt(var(traj_endpoints)), var(traj_endpoints));

    };

    std::vector<int> N {1, 10, 25, 50, 100, 1000};

    printf("n_steps\tavg\tstd\tvar\n");
    printf("------------------------------\n");
    for (auto n : N) {
        fn(n);
    }

    print_vector(simulate_trajectory(100.0, 1));

	return 0;
}

