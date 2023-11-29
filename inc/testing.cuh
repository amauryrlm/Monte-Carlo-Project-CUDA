// Set of tools to test our MC implementations

#pragma once

#include "amaury.cuh"


class SimulationParameters {

public:

    SimulationParameters(
        float volatilty = 0.2,
        float risk_free_rate = 0.1,
        float initial_spot_price = 100.0,
        float contract_strike = 100.0,
        float contract_maturity = 1,
        float barrier = 0
    )
        : sigma{volatilty}
        , r{risk_free_rate}
        , x_0{initial_spot_price}
        , K{contract_strike}
        , T{contract_maturity}
        , B{barrier}
    {}

    float volatility() {
        return this->sigma;
    }

    float risk_free_rate() {
        return this->r;
    }

    float initial_spot_price() {
        return this->x_0;
    }

    float contract_strike() {
        return this->K;
    }

    float contract_maturity() {
        return this->T;
    }

    float barrier() {
        return this->B;
    }


private:

    float sigma;   // volatility
    float r;       // risk-free rate
    float x_0;     // initial_spot_price
    float K;       // contract_strike
    float T;       // contract maturity
    float B;       // barrier

};

// Now, from this class we want to launch different testing suites