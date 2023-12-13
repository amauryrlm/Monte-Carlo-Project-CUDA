#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

#include <cmath>


//cumulative normal distribution function
float CND(float x) {
    float p = 0.2316419f;
    float b1 = 0.31938153f;
    float b2 = -0.356563782f;
    float b3 = 1.781477937f;
    float b4 = -1.821255978f;
    float b5 = 1.330274429f;
    float one_over_twopi = 0.39894228f;
    float t;

    if (x >= 0.0f)
        {
        t = 1.0f / (1.0f + p * x);
        return (1.0f - one_over_twopi * expf(-x * x / 2.0f) * t *
            (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
        }
    else
        {
        t = 1.0f / (1.0f - p * x);
        return (one_over_twopi * expf(-x * x / 2.0f) * t *
            (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
        }
    }


//formula for black scholes option pricing
void black_scholes_CPU(float& call_price, float x0, float strike_price, float T, float risk_free_rate, float volatility)
    {
    float sqrtT = sqrtf(T);
    float    d1 = (logf(x0 / strike_price) + (risk_free_rate + 0.5 * volatility * volatility) * T) / (volatility * sqrtT);
    float    d2 = d1 - volatility * sqrtT;
    float cnd_d1 = CND(d1);
    float cnd_d2 = CND(d2);

    call_price = x0 * cnd_d1 - strike_price * exp(-risk_free_rate * T) * cnd_d2;
    }

#endif // BLACK_SCHOLES_HPP