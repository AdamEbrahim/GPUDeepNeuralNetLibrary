#ifndef SIGMOIDLAYER_H
#define SIGMOIDLAYER_H

#include "Matrix.cuh"
#include "Layer.cuh"

class SigmoidLayer: public Layer {
    protected:
        void callGetActivation(dim3 blocks, dim3 threads, float* w, float* x, float* a, float* b, int xDim, int yDim);
        void callBackPropError(dim3 blocks, dim3 threads, float* nextError, float* w, float* z, float* error, int xDim, int yDim);

    public:
        SigmoidLayer(int prevNumNeurons, int numNeurons);

};

__global__ void getActivation(float* w, float* x, float* a, float* b, int xDim, int yDim);
__global__ void backPropError();

#endif