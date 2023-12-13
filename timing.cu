// Test suite of timing functions to measure the performance of our implementations.
#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include <fstream>

using namespace std::chrono;
using namespace std;

#include "monte_carlo.cuh"

std::vector<float> linspace(const float start, const float end, int n) {
    float diff = (end - start) / (n - 1);
    std::vector<float> out(n);
    out[0] = start;
    out.back() = end;
    for (int i = 1; i < (n - 1); i++) {
        out[i] = out[i - 1] + diff;
    }

    return out;
}


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

    // seconds s() {
    //     return duration_cast<seconds>(stop_ - start_);
    // }

    float ms() {
        return duration_cast<milliseconds>(stop_ - start_).count();
    }

    float microseconds() {
        return duration_cast<std::chrono::microseconds>(stop_ - start_).count();
    }

    float seconds() {
        return microseconds() / (1000.0 * 1000.0);
    }

    void print_elapsed_seconds() {
        printf("Elapsed time: %fs\n", seconds());
    }

    void print_elapsed_microseconds() {
        printf("Elapsed time: %.0f microseconds\n", microseconds());
    }

    float time_fn(function<void()> callback, int n_repetitions = 5) {
        float total_time_s = 0.0;
        for (int i = 0; i < n_repetitions; i++) {
            tic();
            callback();
            toc();
            total_time_s += seconds();
        }

        return total_time_s / n_repetitions;
    }

    float time_fn(function<void(int)> callback, int input, int n_repetitions = 5)  {
        float total_time_s = 0.0;
        for (int i = 0; i < n_repetitions; i++) {
            tic();
            callback(input);
            toc();
            total_time_s += seconds();
        }

        return total_time_s / n_repetitions;
    }

    /**
     * @brief Time a callback function across a range of inputs, storing the output in `csv_file_name`
     *
     * @tparam T
     * @param callback
     * @param input_range
     * @param n_repetitions
     * @param csv_file_name
     * @param input_name
     */
    template <class T>
    void time_fn_range(function<void(T)> callback, std::vector<T> input_range, int n_repetitions = 5, std::string csv_file_name = "timing.csv", std::string input_name = "input") {

        size_t len = input_range.size();
        vector<float> TIMES(len);

        for (int i = 0; i < len; i++) {
            TIMES[i] = time_fn(callback, input_range[i], n_repetitions);
            std::cout << input_name << ": " << input_range[i] << " took: " << TIMES[i] << "s\n";
        }

        // Now write our results to our csv file.
        std::ofstream file;
        file.open(csv_file_name);

        // Add a header
        file << input_name << ",time_s\n";

        for (int i = 0; i < len; i++) {
            file << input_range[i] << "," << TIMES[i] << "\n";
        }

    }

private:

    system_clock::time_point start_;
    system_clock::time_point stop_;

};




template <class T>
void print_vector(vector<T> vec) {

    size_t n = vec.size();

    std::cout << "{";
    for (size_t i = 0; i < n - 1; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec.back() << "}\n";
}


// void cpu_baseline() {

//     Clock clock;
//     int n_traj = 10000;
//     int n_steps = 200;
//     int n_threads_per_block = 10;

//     Simulation parameters(n_traj, n_steps);
//     OptionData option_data = parameters.to_option_data();

//     int n_datapoints = 101; // Number of different values of trajectories
//     int n_trials = 5; // Number of repetitions inside our timing function
//     auto TRAJ = linspace(100, 100100, n_datapoints);

//     for (float f : TRAJ) {
//         std::cout << f << ", ";
//     }
//     printf("\n");
//     // Get a linear range of n_traj values
//     vector<float> durations(n_datapoints);
//     int count = 0;

//     for (float traj : TRAJ) {

//         option_data.N_PATHS = traj;

//         auto cpu_callback = [&] () {
//             auto price = wrapper_cpu_option_vanilla(option_data, n_threads_per_block);
//         };

//         float avg_time = clock.time_fn(cpu_callback, n_trials);
//         printf("[%d] took: %fs\n", (int) traj, avg_time);

//         durations[count] = avg_time;
//         count++;
//     }

//     print_vector(durations);
// }

void gpu_simple_vs_threads() {

    Clock clock;
    int n_traj = 1000000;
    int n_steps = 200;
    int n_threads_per_block = 10;

    Simulation parameters(n_traj, n_steps);
    OptionData option_data = parameters.to_option_data();

    int n_datapoints = 1024; // Number of different values of trajectories
    int n_trials = 5; // Number of repetitions inside our timing function
    auto N_THREADS = linspace(1, 1024, n_datapoints);


    // Get a linear range of n_traj values
    vector<float> durations(n_datapoints);
    int count = 0;

    for (float n_threads: N_THREADS) {

        auto gpu_callback = [&] () {
            auto price = wrapper_gpu_option_vanilla(option_data, n_threads);
        };

        float avg_time = clock.time_fn(gpu_callback, n_trials);
        printf("[%d] took: %fs\n", (int) n_threads, avg_time);

        durations[count] = avg_time;
        count++;
    }

    print_vector(durations);


}

void gpu_simple_vs_threads_compact() {

    Clock clock;
    int n_traj = 100000;
    int n_steps = 200;
    int n_threads_per_block = 10;

    Simulation parameters(n_traj, n_steps);
    OptionData option_data = parameters.to_option_data();

    /* ---------------------------- Timing Parameters --------------------------- */
    std::string output_csv = "gpu_vanilla_time_vs_threads.csv";
    std::string input_name = "n_threads";
    int n_trials = 5;

    std::function<void(float)> gpu_callback = [&] (float n_threads) {
        auto price = wrapper_gpu_option_vanilla(option_data, (int) n_threads);
    };

    auto thread_range = linspace(1, 1024, 1024);

    clock.time_fn_range(
        gpu_callback,
        thread_range,
        n_trials,
        output_csv,
        input_name
    );
}



/**
 * @brief Initialize a new simulation `parameters` and OptionData `option_data` for timing functions.
 *
 * Exports the following symbols:
 * - `Clock clock`
 * - `int n_traj`
 * - `int n_steps`
 * - `int n_threads_per_block`
 * - `Simulation parameters`
 * - `OptionData option_data`
 * - `std::string output_csv`
 * - `std::string input_name
 * - `int n_trials`
 */
#define INIT_SIM(N_TRAJ, N_STEPS, NTPB, OUTPUT_CSV, INPUT_NAME, N_REPETITIONS) \
    Clock clock; \
    int n_traj = N_TRAJ; \
    int n_steps = N_STEPS; \
    int n_threads_per_block = NTPB; \
    \
    Simulation parameters(N_TRAJ, N_STEPS); \
    OptionData option_data = parameters.to_option_data(); \
    \
    std::string output_csv = OUTPUT_CSV; \
    std::string input_name = INPUT_NAME; \
    int n_trials = N_REPETITIONS; \

void cpu_baseline() {

    INIT_SIM(10000, 200, 10, "cpu_vanilla_baseline.csv", "n_traj", 5)

    std::function<void(float)> cpu_callback = [&] (float n_traj) {
        option_data.N_PATHS = n_traj;
        auto price = wrapper_cpu_option_vanilla(option_data, n_threads_per_block);
    };

    auto traj_range = linspace(100, 100100, 101);

    clock.time_fn_range(
        cpu_callback,
        traj_range,
        n_trials,
        output_csv,
        input_name
    );

}

void gpu_baseline() {

    INIT_SIM(10000, 200, 10, "gpu_vanilla_baseline.csv", "n_traj", 5);

    std::function<void(float)> gpu_callback = [&] (float n_traj) {
        option_data.N_PATHS = n_traj;
        auto price = wrapper_gpu_option_vanilla(option_data, n_threads_per_block);
    };

    auto traj_range = linspace(100, 100100, 101);

    clock.time_fn_range(
        gpu_callback,
        traj_range,
        n_trials,
        output_csv,
        input_name
    );

}

void nmc_gpu_baseline() {

    INIT_SIM(10000, 100, 1024, "nmc_baseline_threads.csv", "n_threads", 5);

    const int N_BLOCKS = 10;

    option_data.N_PATHS_INNER = 1000;

    std::function<void(float)> nmc_callback = [&] (float n_threads) {
        auto price = wrapper_gpu_bullet_option_nmc_one_point_one_block(option_data, n_threads, N_BLOCKS);
    };

    auto thread_range = linspace(1, 1024, 1024);

    clock.time_fn_range(
        nmc_callback,
        thread_range,
        n_trials,
        output_csv,
        input_name
    );
}



int main() {

    // first_gpu_simple();
    // gpu_simple_vs_threads();
    // gpu_simple_vs_threads_compact();

    // cpu_baseline();
    // gpu_baseline();
    nmc_gpu_baseline();

}