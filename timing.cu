// Test suite of timing functions to measure the performance of our implementations.
#include <chrono>
using namespace std::chrono;
using namespace std;

#include "monte_carlo.cuh"



class Clock {

public:

    Clock() {
        tic();
        toc();
    }

    void tic() {
        start_ = high_resolution_clock::now();
    }

    void toc() {
        stop_ = high_resolution_clock::now();
    }

    seconds s() {
        return duration_cast<seconds>(stop_ - start_);
    }

    microseconds ms() {
        return duration_cast<microseconds>(stop_ - start_);
    }

private:

    system_clock::time_point start_;
    system_clock::time_point stop_;

};


int main() {



    Clock clock;
    std::cout << "Sup timing!\n";

    clock.tic();






    clock.toc();


    // Let's get down to business
    // Testing the number of blocks vs time it takes


    // Ok the name of the game is benchmarking
    // Start testing time vs the number of steps

    int n_traj = 100;
    int n_steps = 200;
    int n_threads_per_block = 10;

    // All i want to do is simulate using gpu.



}