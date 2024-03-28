#include "SigmoidLayer.cuh"
#include <cmath>

//Kernel functions to be run on GPU


__global__ void getActivation(float* w, float* x, float* a, float* b, int xDim, int yDim) {
    int rowIndexW = (threadIdx.x + blockDim.x * blockIdx.x); //only doing 1D thread blocks because of matrix multiplication
    int stride = gridDim.x * blockDim.x;

    //W*x
    for (int i = rowIndexW; i < yDim; i = i + stride) {
        for (int j = 0; j < xDim; j++) {
            a[i] += w[(i * xDim) + j] * x[j];
        }
        //+b
        a[i] += b[i];
        //Sigmoid(Z)
        a[i] = (1.0 / (1.0 + std::exp(-1.0 * a[i])));
    }

}

void SigmoidLayer::callGetActivation(dim3 blocks, dim3 threads, float* w, float* x, float* a, float* b, int xDim, int yDim) {
    getActivation<<<blocks, threads>>>(w, x, a, b, xDim, yDim);
}


SigmoidLayer::SigmoidLayer(int prevNumNeurons, int numNeurons) : Layer{prevNumNeurons, numNeurons} {

}

void SigmoidLayer::forwardPass(Matrix& prevLayerActivations) {
    float* x = prevLayerActivations.valuesDevice.get(); //no need to cudaMemcpy, updated values will already be on device
    float* w = (this->weights).valuesDevice.get();
    float* b = (this->biases).valuesDevice.get();
    float* a = (this->outputActivation).valuesDevice.get();

    //figure out block/grid dimensions:
    int num_blocks = 1;
    int num_threads = 1;
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);
    
    callGetActivation(blocks, threads, w, x, a, b, this->Weights.xDim, this->Weights.yDim);
    cudaDeviceSynchronize();
}

void SigmoidLayer::backprop(Matrix& nextLayerError, Matrix& nextLayerWeights, Matrix& prevLayerActivations) {

}
