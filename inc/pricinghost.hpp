#include <cmath>

template <typename T>
static T CND(T d)
{
    const T A1 = 0.31938153;
    const T A2 = -0.356563782;
    const T A3 = 1.781477937;
    const T A4 = -1.821255978;
    const T A5 = 1.330274429;
    const T RSQRT2PI = 0.39894228040143267793994605993438;

    T K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    T cnd = RSQRT2PI * exp(-0.5 * d * d) *
            (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

template <typename T>
inline T gaussian_box_muller()
{
    T x = 0.0;
    T y = 0.0;
    T euclid_sq = 0.0;

    do
    {
        x = 2.0 * std::rand() / static_cast<T>(RAND_MAX) - 1;
        y = 2.0 * std::rand() / static_cast<T>(RAND_MAX) - 1;
        euclid_sq = x * x + y * y;
    } while (euclid_sq >= 1.0);

    return x * std::sqrt(-2 * std::log(euclid_sq) / euclid_sq);
}

template <typename T>
inline T BlackScholesHost(
    T r, // risk free rate
    T v, // the volatility
    T s, // initial spot price
    T t, // contract's maturity
    T K  // contract's strike

)
{
    T d1 = (log(s / K) + (r + (v * v) / 2) * t) / (v * std::sqrt(t));
    T d2 = d1 - v * std::sqrt(t);
    T Nd1 = CDN(d1);
    T Nd2 = CDN(d2);

    return Nd1 * s - Nd2 * K * std::exp(-r * t);
}

// template <typename T>
// inline T trajectories( int n_paths,
//                        int n_steps,
//                        T step,
//                        T s,
//                        T t)
// {

//     T g = gaussian_box_muller();
//     return s * expf((r - v * v / 2) * step + v * sqrtf(step) * g);
// }
