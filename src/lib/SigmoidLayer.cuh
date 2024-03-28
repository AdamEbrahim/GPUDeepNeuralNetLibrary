#ifndef SIGMOIDLAYER_H
#define SIGMOIDLAYER_H

#include "Matrix.cuh"
#include "Layer.cuh"

class SigmoidLayer: public Layer {
    private:
        void callGetActivation(dim3 blocks, dim3 threads, float* w, float* x, float* a, float* b, int xDim, int yDim);

    public:
        SigmoidLayer(int prevNumNeurons, int numNeurons);
        void forwardPass(Matrix& prevLayerActivations);
        void backprop(Matrix& nextLayerError, Matrix& nextLayerWeights, Matrix& prevLayerActivations);

};

__global__ void getActivation(float* w, float* x, float* a, float* b, int xDim, int yDim);

#endif