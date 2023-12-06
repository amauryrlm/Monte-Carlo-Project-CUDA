#include <math.h>
#include <iostream>
using namespace std;



void simulateOptionPriceCPU(float* optionPriceCPU, int N_PATHS, int N_STEPS, float* h_randomData, float S0, float sigma, float sqrdt, float r, float K, float dt, float* simulated_paths_cpu) {
    float G;
    float countt = 0.0f;
    for (int i = 0; i < N_PATHS;i++) {
        float St = S0;
        for (int j = 0; j < N_STEPS; j++) {
            G = h_randomData[i * N_STEPS + j];
            St *= expf((r - (sigma * sigma) / 2) * dt + sigma * sqrdt * G);

            }

        simulated_paths_cpu[i] = max(St - K, 0.0f);
        // cout << "cpu : " <<  St << endl;
        countt += max(St - K, 0.0f);
        }
    *optionPriceCPU = expf(-r) * (countt / N_PATHS);
    }