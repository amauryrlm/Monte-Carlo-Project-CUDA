/**========================================================================
 * ?                          common.hpp
 * @brief   : Common utility functions
 * @details :
 * @author  : Evan Voyles
 * @email   : ejovo13@yahoo.com
 * @date    : 2023-10-26
 *========================================================================**/
#pragma once

#include <iostream>
#include <vector>

namespace monte_carlo {

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

}; // namespace monte_c carlo