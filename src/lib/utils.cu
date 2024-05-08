#include "utils.cuh"
#include <cmath>

__device__ float sigmoid(float z) {
    return (1.0 / (1.0 + std::exp(-1.0 * z)));
}

__device__ float sigmoidPrime(float z) {
    float a = (1.0 + std::exp(-1.0 * z))
    return (std::exp(-1.0 * z)) / (pow(a, 2));
}