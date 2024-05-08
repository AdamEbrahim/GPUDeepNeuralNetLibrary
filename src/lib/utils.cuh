#ifndef UTILS_H
#define UTILS_H

//device code; only callable by global or other device functions
__device__ float sigmoid(float z);
__device__ float sigmoidPrime(float z);

#endif