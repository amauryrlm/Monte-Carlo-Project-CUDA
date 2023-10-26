/**========================================================================
 * ?                          trajectories.cpp
 * @brief   : Compute Monte Carlo trajectories for spot prices S
 * @details :
 * @author  : Evan Voyles
 * @email   : ejovo13@yahoo.com
 * @date    : 2023-10-26
 *========================================================================**/
#pragma once

#include <cmath>
#include <vector>
#include <functional>

#include "Xoshiro.hpp"

#define DEFAULT_VOLATILITY 0.2
#define DEFAULT_RISK_FREE_RATE 0.1
#define DEFAULT_INITIAL_SPOT_PRICE 100
#define DEFAULT_END_TIME 1.0

namespace monte_carlo {


/// @brief Simulate $S_{t + \\delta t}$
/// @tparam Float Either double or float. Allows us to compare the speed of 64 and 32-bit arithmetic
/// @param current_spot_price
/// @param time_step
/// @param v Volatility
/// @param r Risk-free rate
/// @return
template<class Float>
inline auto next_spot_price(
    Float current_spot_price,
    Float time_step,
    Float v = DEFAULT_VOLATILITY,
    Float r = DEFAULT_RISK_FREE_RATE
) -> Float {

    using namespace ejovo::rng;

    // Get a normal random variate
    Float g = box_muller();
    return current_spot_price * std::exp((r - v * v * 0.5) * time_step + v * std::sqrt(time_step) * g);
}

/// @brief Simulate a trajectory of spot prices from [0, T] using `n` steps.
/// This function does not store the intermediate spot prices. To simulate and store the entire
/// trajectory, use `simulate_trajectory`
/// @tparam Float
/// @param x0
/// @param v
/// @param T
/// @param n_steps The number of time steps to simulate.
/// @return
template<class Float>
inline auto compute_trajectory_endpoint (
    Float x0,
    const int n_steps = 100,
    const Float T = DEFAULT_END_TIME,
    Float v = DEFAULT_VOLATILITY,
    Float r = DEFAULT_RISK_FREE_RATE
) -> Float {

    const Float time_step = T / n_steps;
    Float spot_price = x0;

    for (int i = 0; i < n_steps; i++) {
        spot_price = next_spot_price(spot_price, time_step, v, r);
    }

    return spot_price;
}

template<class Float>
inline auto simulate_trajectory (
    Float x0,
    const int n_steps = 100,
    const Float T = DEFAULT_END_TIME,
    Float v = DEFAULT_VOLATILITY,
    Float r = DEFAULT_RISK_FREE_RATE
) -> std::vector<Float> {

    const Float time_step = T / n_steps;
    std::vector<Float> trajectory {};
    trajectory.push_back(x0);

    for (int i = 0; i < n_steps; i++) {
        trajectory.push_back(next_spot_price(trajectory[i], time_step, v, r));
    }

    return trajectory;
}

template<class T>
inline auto replicate (std::function<T()> fn, int n) -> std::vector<T> {
    std::vector<T> out {};
    for (int i = 0; i < n; i++) {
        out.push_back(fn());
    }
    return out;
}

template<class Float>
inline auto compute_n_trajectory_endpoints (
    Float x0,
    const int n_trajectories = 100,
    const int n_steps = 100,
    const Float T = DEFAULT_END_TIME,
    Float v = DEFAULT_VOLATILITY,
    Float r = DEFAULT_RISK_FREE_RATE
) -> std::vector<Float> {

    // std::cout << "Computing trajectories with " << n_steps << " steps\n";

    auto fn = [&] () {
        return compute_trajectory_endpoint(x0, n_steps, T, v, r);
    };

    return replicate<Float>(fn, n_trajectories);
}

}; // namespace monte_carlo