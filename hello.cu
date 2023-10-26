#include <iostream>
// #include <format>
// #include <functional>
#include <cuda_runtime.h>


#include "trajectories.hpp"
#include "Xoshiro.hpp"


template<class T>
void print_vector(std::vector<T> vec) {
    const auto n = vec.size();

    std::cout << "{";
    for (int i = 0; i < n - 1; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec.back() << "}\n";
}

template<class Float>
inline Float sum(std::vector<Float> vec) {
    Float total = 0;
    for (auto &el : vec) {
        total += el;
    }
    return total;
}

template<class Float>
inline Float mean(std::vector<Float> vec) {
    return sum(vec) / vec.size();
}

template<class Float>
inline Float var(std::vector<Float> vec, bool population = true) {

    Float mu = mean(vec);
    Float out = 0;
    const auto n = vec.size();

    for (int i = 0; i < n; i++) {
        const Float a = (vec[i] - mu);
        out += a * a;
    }

    if (population) return out / n;
    else return out / (n - 1);

}

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

    printf("n_steps\tavg\tstd\t\tvar\n");
    for (auto n : N) {
        fn(n);
    }






	return 0;
}

