#include "SigmoidLayer.cuh"

SigmoidLayer::SigmoidLayer(int prevNumNeurons, int numNeurons) : Layer{prevNumNeurons, numNeurons} {
    numberNeurons = numNeurons;
    weights.allocateHostMemory();
    weights.allocateCUDAMemory();

    biases.allocateHostMemory();
    biases.allocateCUDAMemory();

    outputActivation.allocateHostMemory();
    outputActivation.allocateCUDAMemory();

    inputError.allocateHostMemory();
    inputError.allocateCUDAMemory();
}

void SigmoidLayer::forwardPass(Matrix& prevLayerActivations) {

}