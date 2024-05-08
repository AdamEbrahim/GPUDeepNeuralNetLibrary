#include "SigmoidLayer.cuh"
#include "Matrix.cuh"
#include "utils.cuh"
#include <cmath>

//Kernel and device functions to be run on GPU

__global__ void getActivation(float* w, float* x, float* a, float* b, float* z, int xDim, int yDim) {
    int rowIndexW = (threadIdx.x + blockDim.x * blockIdx.x); //only doing 1D thread blocks because of matrix multiplication
    int stride = gridDim.x * blockDim.x;

    //W*x
    for (int i = rowIndexW; i < yDim; i = i + stride) {
        a[i] = 0; //reinitialize to 0
        for (int j = 0; j < xDim; j++) {
            a[i] += w[(i * xDim) + j] * x[j];
        }
        //+b
        a[i] += b[i];
        //Sigmoid(Z)
        z[i] = a[i]; //set weighted input matrix before applying activation function
        a[i] = sigmoid(a[i]);
    }

}

//used for backpropagation
__global__ void backPropError(float* nextError, float* w, float* z, float* error, int xDim, int yDim) {
    int rowIndex = threadIdx.x + (blockIdx.x * blockDim.x);
    int stride = blockDim.x * gridDim.x;

    //i < xDim and j < yDim because multiplying transpose of next layer's weight matrix
    for (int i = rowIndex; i < xDim; i = i + stride) {
        error[i] = 0; //reinitialize to 0
        for (int j = 0; j < yDim; j++) {
            error[i] += w[(j * xDim) + i] * nextError[j];
        }
        //hadamard product with sigmoid prime w.r.t weighted input
        error[i] = error[i] * sigmoidPrime(z[i]);
    }
}

//constructor
SigmoidLayer::SigmoidLayer(int prevNumNeurons, int numNeurons) : Layer{prevNumNeurons, numNeurons} {

}

void SigmoidLayer::callGetActivation(dim3 blocks, dim3 threads, float* w, float* x, float* a, float* b, float* z, int xDim, int yDim) {
    getActivation<<<blocks, threads>>>(w, x, a, b, z, xDim, yDim);
}

void SigmoidLayer::callBackPropError(dim3 blocks, dim3 threads, float* nextError, float* w, float* z, float* error, int xDim, int yDim) {
    backPropError<<<blocks, threads>>>(nextError, w, z, error, xDim, yDim);
}
